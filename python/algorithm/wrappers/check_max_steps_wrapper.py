from __future__ import annotations

from typing import SupportsFloat, Any

import numpy as np
from ap_gym import (
    ActivePerceptionVectorWrapper,
    BaseActivePerceptionEnv,
    BaseActivePerceptionVectorEnv,
    ActivePerceptionWrapper,
)


class CheckMaxStepsWrapper(ActivePerceptionWrapper):
    def __init__(self, env: BaseActivePerceptionEnv, step_limit: int):
        super().__init__(env)
        self.__current_steps = 0
        self.__step_limit = step_limit

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.__current_steps += 1
        if self.__current_steps > self.__step_limit:
            raise RuntimeError(
                f"Environment surpassed step limit ({self.__step_limit})."
            )
        obs, reward, done, trunc, info = super().step(action)
        if done or trunc:
            self.__current_steps = -1  # Next step will be a reset
        return obs, reward, done, trunc, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.__current_steps = 0
        return obs, info


class CheckMaxStepsVectorWrapper(ActivePerceptionVectorWrapper):
    def __init__(self, env: BaseActivePerceptionVectorEnv, step_limit: int):
        super().__init__(env)
        self.__current_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.__step_limit = step_limit

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.__current_steps += 1
        if np.any(self.__current_steps > self.__step_limit):
            raise RuntimeError(
                f"Environment surpassed step limit ({self.__step_limit})."
            )
        obs, reward, done, trunc, info = super().step(action)
        self.__current_steps[done | trunc] = -1  # Next step will be a reset
        return obs, reward, done, trunc, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        self.__current_steps = np.zeros(self.num_envs, dtype=np.int32)
        return obs, info
