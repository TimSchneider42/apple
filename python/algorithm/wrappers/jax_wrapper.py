from __future__ import annotations

import copy
from typing import Any, Generic, TypeVar

import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from ap_gym import ActivePerceptionVectorWrapper, ImageSpace
from gym_space_map import gym_space_map
from hostcallback import io_callback, DataTransferMode
from util import fix_ordered_dicts
from .log_wrapper import LogWrapper

ArrayType = TypeVar("ArrayType")


@flax.struct.dataclass
class ActivePerceptionMetadata(Generic[ArrayType]):
    base_reward: ArrayType
    perception_target: Any


@flax.struct.dataclass
class EnvStepReturn(Generic[ArrayType]):
    obs: Any
    reward: ArrayType
    terminated: ArrayType
    truncated: ArrayType
    metadata: ActivePerceptionMetadata
    total_environment_steps: int


@flax.struct.dataclass
class EnvResetReturn(Generic[ArrayType]):
    obs: Any
    total_environment_steps: int


class JaxWrapper(ActivePerceptionVectorWrapper):
    def __init__(
        self,
        env: LogWrapper,
        max_img_size: tuple[int, int] | None = None,
        img_target_dtype: jnp.dtype | None = None,
    ):
        super().__init__(env)
        self.__img_target_dtype = img_target_dtype
        if self.__img_target_dtype is not None:
            assert jnp.issubdtype(
                self.__img_target_dtype, jnp.integer
            ) or jnp.issubdtype(self.__img_target_dtype, jnp.floating)
        obs_result_shape = gym_space_map(
            lambda x: jax.ShapeDtypeStruct((env.num_envs, *x.shape), x.dtype),
            env.single_observation_space,
        )
        act_result_shape = gym_space_map(
            lambda x: jax.ShapeDtypeStruct((env.num_envs, *x.shape), x.dtype),
            env.single_action_space,
        )

        prediction_target_space = gym_space_map(
            lambda x: x, self.single_prediction_target_space
        )
        self.__perception_metadata_shape_single = ActivePerceptionMetadata(
            jax.ShapeDtypeStruct((), jnp.float32),
            jax.tree.map(
                lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype),
                prediction_target_space,
            ),
        )
        metadata_shape = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                (
                    env.num_envs,
                    *x.shape,
                ),
                x.dtype,
            ),
            self.__perception_metadata_shape_single,
        )

        result_shape_step = EnvStepReturn(
            obs_result_shape,
            jax.ShapeDtypeStruct((env.num_envs,), jnp.float32),
            jax.ShapeDtypeStruct((env.num_envs,), jnp.bool_),
            jax.ShapeDtypeStruct((env.num_envs,), jnp.bool_),
            metadata_shape,
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        self.__step_jax_hcb = io_callback(
            result_shape=result_shape_step,
            data_transfer_mode_host_to_device=DataTransferMode.PACKED,
            data_transfer_mode_device_to_host=DataTransferMode.PACKED,
        )(self.__step_jax)

        result_shape_reset = EnvResetReturn(
            obs_result_shape, jax.ShapeDtypeStruct((), jnp.int32)
        )
        self.__reset_jax_hcb = io_callback(
            result_shape=result_shape_reset,
            data_transfer_mode_host_to_device=DataTransferMode.PACKED,
            data_transfer_mode_device_to_host=DataTransferMode.PACKED,
        )(self.__reset_jax)

        self.__sample_random_action_hcb = io_callback(
            result_shape=act_result_shape,
            data_transfer_mode_host_to_device=DataTransferMode.PACKED,
            data_transfer_mode_device_to_host=DataTransferMode.PACKED,
        )(self.__sample_random_action)

        self.__max_img_size = max_img_size
        self._observation_space = gym_space_map(
            self.__transform_space, env.observation_space, return_gym_space=True
        )
        self.__obs_space_tree = gym_space_map(lambda x: x, self._observation_space)
        self.__orig_obs_space_tree = gym_space_map(lambda x: x, env.observation_space)
        self._single_observation_space = gym_space_map(
            self.__transform_space, env.single_observation_space, return_gym_space=True
        )

    def __transform_space(self, space: gym.Space) -> gym.Space:
        if isinstance(space, ImageSpace) and self.__max_img_size is not None:
            if self.__img_target_dtype is not None:
                img_dtype = self.__img_target_dtype
                if jnp.issubdtype(img_dtype, jnp.floating):
                    low = 0.0
                    high = 1.0
                else:
                    low = jnp.iinfo(img_dtype).min
                    high = jnp.iinfo(img_dtype).max
            else:
                img_dtype = space.dtype
                low = space.low
                high = space.high
            return ImageSpace(
                min(self.__max_img_size[0], space.width),
                min(self.__max_img_size[1], space.height),
                space.channels,
                batch_shape=space.batch_shape,
                dtype=img_dtype,
                seed=space._np_random,
                low=low,
                high=high,
            )
        return space

    @staticmethod
    def __resize_and_convert(
        img: jax.Array, space: ImageSpace, orig_space: ImageSpace
    ) -> jax.Array:
        img = (img - orig_space.low) / (orig_space.high - orig_space.low)
        if space.shape != orig_space.shape:
            img = jax.image.resize(
                img, space.shape, method="bicubic", antialias=True
            ).clip(0, 1)
        return (img * (space.high - space.low) + space.low).astype(space.dtype)

    def __transform_img(
        self, x: Any, space: gym.Space, orig_space: gym.Space
    ) -> np.ndarray:
        if isinstance(space, ImageSpace):
            assert isinstance(orig_space, ImageSpace)
            if (
                space.dtype != orig_space.dtype
                or space.shape != orig_space.shape
                or np.any(space.low != orig_space.low)
                or np.any(space.high != orig_space.high)
            ):
                if (
                    jnp.issubdtype(orig_space.dtype, jnp.integer)
                    and space.dtype != orig_space.dtype
                ):
                    x = x.astype(
                        space.dtype
                        if jnp.issubdtype(space.dtype, jnp.floating)
                        else jnp.float32
                    )
                x = self.__resize_and_convert(x, space, orig_space)
        return x

    def step(
        self, actions: gym.core.ActType
    ) -> tuple[gym.core.ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(actions)
        return fix_ordered_dicts(obs), reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return fix_ordered_dicts(obs), info

    @classmethod
    def __remove_masks(cls, perception_target: Any):
        def remove_mask_dict(x: Any):
            if isinstance(x, dict):
                return {
                    k: cls.__remove_masks(v)
                    for k, v in x.items()
                    if not k.startswith("_")
                }
            return x

        return jax.tree.map(
            remove_mask_dict, perception_target, is_leaf=lambda x: isinstance(x, dict)
        )

    def __get_perception_metadata(
        self, info: dict[str, Any]
    ) -> ActivePerceptionMetadata:
        base_reward = info.get("base_reward", np.zeros(self.num_envs, dtype=np.float32))
        if "prediction" not in info:
            # All envs are resetting at the same time. In this case just fill in zeros (it will be ignored higher up
            # anyway)
            target = jax.tree.map(
                lambda x: np.zeros_like(x), self.prediction_target_space.sample()
            )
        else:
            target = jax.tree.map(
                lambda x: self.__remove_masks(x["target"]),
                info["prediction"],
                is_leaf=lambda x: isinstance(x, dict) and "target" in x,
            )
        return ActivePerceptionMetadata(base_reward, target)

    def __step_jax(self, action: Any, prediction: Any) -> EnvStepReturn[np.ndarray]:
        full_action = {"action": action, "prediction": prediction}
        obs, reward, terminated, truncated, info = self.step(
            jax.tree.map(np.array, full_action)
        )

        assert isinstance(self.env, LogWrapper)
        return EnvStepReturn(
            obs,
            reward,
            terminated,
            truncated,
            self.__get_perception_metadata(info),
            self.env.total_environment_steps,
        )

    def __reset_jax(self, seed: jax.Array | None) -> EnvResetReturn[np.ndarray]:
        obs, info = self.reset(seed=None if seed is None else seed.item())
        assert isinstance(self.env, LogWrapper)
        return EnvResetReturn(obs, self.env.total_environment_steps)

    def step_jax(self, action: Any, prediction: Any) -> EnvStepReturn[jax.Array]:
        env_step_return: EnvStepReturn[jax.Array] = self.__step_jax_hcb(
            action, prediction
        )
        env_step_return = env_step_return.replace(
            obs=jax.tree.map(
                self.__transform_img,
                fix_ordered_dicts(env_step_return.obs),
                self.__obs_space_tree,
                self.__orig_obs_space_tree,
            )
        )
        return env_step_return

    def reset_jax(self, seed: jax.Array) -> EnvResetReturn[jax.Array]:
        env_reset_return: EnvResetReturn[jax.Array] = self.__reset_jax_hcb(seed)
        env_reset_return = env_reset_return.replace(
            obs=jax.tree.map(
                self.__transform_img,
                fix_ordered_dicts(env_reset_return.obs),
                self.__obs_space_tree,
                self.__orig_obs_space_tree,
            )
        )
        return env_reset_return

    def __sample_random_action(self, seed: jax.Array):
        action_space = copy.deepcopy(self.action_space)
        action_space.seed(seed.item())
        return action_space.sample()

    def sample_random_action(self, rng: jax.Array):
        return self.__sample_random_action_hcb(
            jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max)
        )

    @property
    def perception_metadata_shape_single(self):
        return self.__perception_metadata_shape_single
