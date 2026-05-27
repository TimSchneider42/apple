# Environment used in
# Fleer, S., Moringen, A., Klatzky, R. L., & Ritter, H. (2020). Learning efficient haptic shape exploration with a rigid
# tactile sensor array. PloS one, 15(1), e0226880.
#
# ham_tactile_classification_objects.hdf5 is licensed under GNU GENERAL PUBLIC LICENSE and directly copied from
# https://github.com/fleer/Haptic-Attention-Model, where it is called objects.hdf5.
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import h5py
import numpy as np
from PIL import Image

from ap_gym import ActiveClassificationEnv


@functools.cache
def load_data() -> dict[str, np.ndarray]:
    data_file = h5py.File(
        Path(__file__).parent / "ham_tactile_classification_objects.hdf5"
    )
    return {
        k: v[:].reshape((*v.shape[0:2], 16, 16)).astype(np.float32)
        for k, v in data_file.items()
    }


class HAMTactileClassificationEnv(ActiveClassificationEnv[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: Literal["rgb_array"] = "rgb_array"):
        self.__data = load_data()
        self.__label_id_mapping = {
            i: k for i, k in enumerate(sorted(self.__data.keys()))
        }
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Invalid render mode: {render_mode}")
        self.__render_mode = render_mode

        super().__init__(
            len(self.__label_id_mapping),
            gym.spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32,
            ),
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=self.__data[self.__label_id_mapping[0]].shape[-2:],
            dtype=np.float32,
        )
        self.__last_obs: np.ndarray | None = None
        self.__current_label: int | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any | None] = None):
        super().reset(seed=seed, options=options)
        self.__current_label = self.np_random.integers(0, len(self.__label_id_mapping))
        self.__last_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return self.__last_obs, {}

    def _step(self, action: np.ndarray, prediction: np.ndarray):
        action = action.clip(-1, 1)
        pos, angle = action
        obj = self.__data[self.__label_id_mapping[self.__current_label]]
        index_l = int((1 + pos) * (obj.shape[0] - 1) / 2)
        index_e = int((1 + angle) * (obj.shape[1] - 1) / 2)
        self.__last_obs = obj[index_l, index_e]
        return self.__last_obs, 0.0, False, False, {}, self.__current_label

    def render(self):
        return np.array(
            Image.fromarray(self.__last_obs * 255)
            .resize((128, 128), resample=Image.NEAREST)
            .convert("RGB")
        )

    @property
    def render_mode(self) -> Literal["rgb_array"]:
        return self.__render_mode
