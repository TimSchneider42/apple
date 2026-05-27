from typing import Generic, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

import ap_gym
from ap_gym import ActivePerceptionVectorWrapper
from ap_gym.types import (
    ObsType,
    ActType,
    PredType,
    PredTargetType,
    ArrayType,
    FullActType,
)


class AddRenderObservationVectorWrapper(
    ActivePerceptionVectorWrapper[
        Dict[str, ObsType | np.ndarray] | np.ndarray,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
        ObsType,
        ActType,
        PredType,
        PredTargetType,
        ArrayType,
    ],
    Generic[ObsType, ActType, PredType, PredTargetType, ArrayType],
):
    def __init__(
        self,
        env: gym.vector.VectorEnv[ObsType, FullActType[ActType, PredType], ArrayType],
        render_only: bool = True,
        render_key: str = "pixels",
        obs_key: str = "state",
        target_img_dtype: np.dtype | None = None,
    ):
        super().__init__(env)

        self.__obs_key = obs_key
        self.__render_key = render_key
        self.__render_only = render_only

        env.reset(seed=0)
        sample_img = env.render()[0]
        height, width, channels = sample_img.shape

        if target_img_dtype is None:
            target_img_dtype = sample_img.dtype
        self.__target_img_dtype = target_img_dtype

        if target_img_dtype == np.uint8:
            img_dtype = np.uint8
            low = 0
            high = 255
        elif np.issubdtype(target_img_dtype, np.floating):
            img_dtype = np.float32
            low = 0.0
            high = 1.0
        else:
            raise ValueError(
                f"Unsupported target_img_dtype: {target_img_dtype}. "
                "Supported dtypes are np.uint8 and np.floating."
            )

        pixel_space = ap_gym.ImageSpace(
            width,
            height,
            channels,
            dtype=img_dtype,
            low=low,
            high=high,
        )
        if self.__render_only:
            self._single_observation_space = pixel_space
        else:
            if isinstance(env.observation_space, gym.spaces.Dict):
                single_obs_space_dict = env.single_observation_space.spaces.copy()
            else:
                single_obs_space_dict = {self.__obs_key: env.single_observation_space}
            single_obs_space_dict[self.__render_key] = pixel_space
            self._single_observation_space = gym.spaces.Dict(single_obs_space_dict)
        self._observation_space = batch_space(
            self._single_observation_space, self.num_envs
        )

    def make_obs(self, obs: ObsType) -> Dict[str, ObsType | np.ndarray] | np.ndarray:
        rendered_imgs = np.asarray(self.env.render())

        if rendered_imgs.dtype != self.__target_img_dtype:
            if np.issubdtype(rendered_imgs.dtype, np.floating):
                if np.issubdtype(self.__target_img_dtype, np.floating):
                    rendered_imgs = rendered_imgs.astype(self.__target_img_dtype)
                else:
                    rendered_imgs = (rendered_imgs * 255.0).astype(
                        self.__target_img_dtype
                    )
            else:
                rendered_imgs = np.round(rendered_imgs * 255.0).astype(np.uint8)

        if self.__render_only:
            return rendered_imgs
        else:
            if isinstance(obs, dict):
                return {**obs, self.__render_key: rendered_imgs}
            else:
                return {self.__obs_key: obs, self.__render_key: rendered_imgs}

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.make_obs(obs), rewards, terminations, truncations, infos

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.make_obs(obs), info
