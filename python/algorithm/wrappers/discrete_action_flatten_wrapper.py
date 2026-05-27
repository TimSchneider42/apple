from __future__ import annotations

import math
from typing import SupportsFloat, Any

import gymnasium
import jax
import numpy as np

from gym_space_map import gym_space_map


def _convert_space(space: gymnasium.Space) -> gymnasium.Space:
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        if any(s != 0 for s in space.start):
            raise NotImplementedError(
                f"Only start == 0 is supported, not {space.start}."
            )
        return gymnasium.spaces.Discrete(math.prod(space.nvec), seed=space._np_random)
    return space


def _convert_value(value: Any, space: gymnasium.Space) -> Any:
    if isinstance(space, gymnasium.spaces.MultiDiscrete):
        return np.stack(np.unravel_index(value, space.nvec), axis=-1)
    return value


def _convert(value: Any, space: gymnasium.Space) -> Any:
    return jax.tree.map(_convert_value, value, space)


class DiscreteActionFlattenWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._action_space = gym_space_map(
            _convert_space, env.action_space, return_gym_space=True
        )
        self.__inner_act_space_tree = gym_space_map(lambda x: x, env.action_space)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = _convert(action, self.__inner_act_space_tree)
        return super().step(action)


class DiscreteActionFlattenVectorWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, env: gymnasium.vector.VectorEnv):
        super().__init__(env)
        self._action_space = gym_space_map(
            _convert_space, env.action_space, return_gym_space=True
        )
        self.__inner_act_space_tree = gym_space_map(lambda x: x, env.action_space)
        self._single_action_space = gym_space_map(
            _convert_space, env.single_action_space, return_gym_space=True
        )

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        action = _convert(action, self.__inner_act_space_tree)
        return super().step(action)
