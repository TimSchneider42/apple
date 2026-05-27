from __future__ import annotations

import copy
from typing import SupportsFloat, Any

import numpy as np

from ap_gym import (
    ActivePerceptionVectorWrapper,
    BaseActivePerceptionVectorEnv,
)


class ClassificationBinaryRewardVectorWrapper(ActivePerceptionVectorWrapper):
    def __init__(self, env: BaseActivePerceptionVectorEnv):
        super().__init__(env)
        self.__use = False

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, trunc, info = super().step(action)

        if "base_reward" in info:
            _prediction = info.get(
                "_prediction", np.ones((self.num_envs,), dtype=np.bool_)
            )
            _base_reward = info.get(
                "_base_reward", np.ones((self.num_envs,), dtype=np.bool_)
            )
            _target = info["prediction"].get(
                "_target", np.ones((self.num_envs,), dtype=np.bool_)
            )

            assert np.all(_prediction == _base_reward)
            assert np.all(_target == _base_reward)

            assert np.all(reward[~_base_reward] == 0.0)

            prediction = action["prediction"][_base_reward]
            target = info["prediction"]["target"][_base_reward]
            correct = np.argmax(prediction, axis=-1) == target

            reward = np.zeros_like(reward)
            reward[_base_reward] = info["base_reward"][_base_reward] + correct.astype(
                np.float32
            )
        else:
            assert np.all(reward == 0.0)

        return obs, reward, done, trunc, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        return obs, info
