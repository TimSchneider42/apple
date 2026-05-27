from __future__ import annotations

import logging
from typing import SupportsFloat, Any, overload, Type

import gymnasium
import jax
import numpy as np

from algorithm.wrappers.discrete32 import Discrete32
from ap_gym import (
    ActivePerceptionWrapper,
    BaseActivePerceptionEnv,
    BaseActivePerceptionVectorEnv,
    ActivePerceptionVectorWrapper,
    ImageSpace,
    LogitSpace,
)
from gym_space_map import gym_space_map

logger = logging.getLogger(__name__)


def full_name(object_type: Type[Any]) -> str:
    if object_type.__module__ == "__builtin__":
        return object_type.__name__
    return object_type.__module__ + "." + object_type.__name__


def _convert_space(space: gymnasium.Space) -> gymnasium.Space:
    if isinstance(space, gymnasium.spaces.Discrete):
        output_space = Discrete32(space.n, seed=space._np_random, start=space.start)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        output_space = gymnasium.spaces.MultiDiscrete(
            space.nvec, dtype=np.int32, seed=space._np_random, start=space.start
        )
    elif isinstance(space, gymnasium.spaces.Box):
        if np.issubdtype(space.dtype, np.integer):
            dtype = np.int32
        elif np.issubdtype(space.dtype, np.floating):
            dtype = np.float32
        else:
            raise NotImplementedError(f"Unsupported dtype {space.dtype}")
        if isinstance(space, ImageSpace):
            output_space = ImageSpace(
                space.width,
                space.height,
                space.channels,
                batch_shape=space.batch_shape,
                dtype=dtype,
                seed=space._np_random,
                low=space.low,
                high=space.high,
            )
        elif isinstance(space, LogitSpace):
            return LogitSpace(
                space.low, space.high, space.shape, dtype=dtype, seed=space._np_random
            )
        else:
            output_space = gymnasium.spaces.Box(
                space.low, space.high, space.shape, dtype=dtype, seed=space._np_random
            )
    else:
        raise NotImplementedError(f"Unsupported space {space}")

    if (
        type(output_space) is not type(space)
        and type(space) is not gymnasium.spaces.Discrete
    ):
        logger.warning(
            f"GymWrapper32 generalized unknown specialization "
            f'"{full_name(type(space))}" to "{full_name(type(output_space))}".'
        )

    return output_space


def _convert(value: Any, space: gymnasium.Space) -> Any:
    return jax.tree.map(lambda x, s: x.astype(s.dtype), value, space)


def _to32(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == np.int64:
            return value.astype(np.int32)
        elif value.dtype == np.float64:
            return value.astype(np.float32)
    return value


class GymSingleWrapper32(ActivePerceptionWrapper):
    def __init__(self, env: BaseActivePerceptionEnv):
        super().__init__(env)
        self._action_space = gym_space_map(
            _convert_space, env.action_space, return_gym_space=True
        )
        self._observation_space = gym_space_map(
            _convert_space, env.observation_space, return_gym_space=True
        )
        self._prediction_target_space = gym_space_map(
            _convert_space, env.prediction_target_space, return_gym_space=True
        )
        self.__inner_act_space_tree = gym_space_map(lambda x: x, env.action_space)
        self.__obs_space_tree = gym_space_map(lambda x: x, self._observation_space)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = _convert(action, self.__inner_act_space_tree)
        obs, reward, done, trunc, info = super().step(action)
        return _convert(obs, self.__obs_space_tree), reward, done, trunc, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return _convert(obs, self.__obs_space_tree), info


class GymVectorWrapper32(ActivePerceptionVectorWrapper):
    def __init__(self, env: BaseActivePerceptionVectorEnv):
        super().__init__(env)
        self._action_space = gym_space_map(
            _convert_space, env.action_space, return_gym_space=True
        )
        self._observation_space = gym_space_map(
            _convert_space, env.observation_space, return_gym_space=True
        )
        self._prediction_target_space = gym_space_map(
            _convert_space, env.prediction_target_space, return_gym_space=True
        )
        self.__inner_act_space_tree = gym_space_map(lambda x: x, env.action_space)
        self.__obs_space_tree = gym_space_map(lambda x: x, self._observation_space)
        self._single_observation_space = gym_space_map(
            _convert_space, env.single_observation_space, return_gym_space=True
        )
        self._single_action_space = gym_space_map(
            _convert_space, env.single_action_space, return_gym_space=True
        )
        self._single_prediction_target_space = gym_space_map(
            _convert_space, env.single_prediction_target_space, return_gym_space=True
        )

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = _convert(action, self.__inner_act_space_tree)
        obs, reward, done, trunc, info = super().step(action)
        return (
            _convert(obs, self.__obs_space_tree),
            reward.astype(np.float32),
            done,
            trunc,
            jax.tree.map(_to32, info),
        )

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return _convert(obs, self.__obs_space_tree), jax.tree.map(_to32, info)


@overload
def GymWrapper32(env: BaseActivePerceptionEnv) -> GymSingleWrapper32: ...


@overload
def GymWrapper32(env: BaseActivePerceptionVectorEnv) -> GymVectorWrapper32: ...


def GymWrapper32(
    env: BaseActivePerceptionEnv | BaseActivePerceptionVectorEnv,
) -> GymSingleWrapper32 | GymVectorWrapper32:
    if isinstance(env, BaseActivePerceptionVectorEnv):
        return GymVectorWrapper32(env)
    elif isinstance(env, BaseActivePerceptionEnv):
        return GymSingleWrapper32(env)
    else:
        raise NotImplementedError(f"Unsupported environment {env}")
