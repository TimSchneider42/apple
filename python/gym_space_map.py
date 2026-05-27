from __future__ import annotations

from typing import Callable, Any

import gymnasium as gym
from ap_gym import ActivePerceptionActionSpace


def gym_space_map(
    f: Callable[[gym.Space], Any], space: gym.Space, return_gym_space: bool = False
) -> Any:
    if return_gym_space and isinstance(space, ActivePerceptionActionSpace):
        return ActivePerceptionActionSpace(
            gym_space_map(f, space.inner_action_space, return_gym_space=True),
            gym_space_map(f, space.prediction_space, return_gym_space=True),
        )
    elif isinstance(space, gym.spaces.Dict):
        d = {
            k: gym_space_map(f, v, return_gym_space=return_gym_space)
            for k, v in space.spaces.items()
        }
        return gym.spaces.Dict(d, seed=space._np_random) if return_gym_space else d
    elif isinstance(space, gym.spaces.Tuple):
        t = tuple(
            gym_space_map(f, v, return_gym_space=return_gym_space) for v in space.spaces
        )
        return gym.spaces.Tuple(t, seed=space._np_random) if return_gym_space else t
    else:
        return f(space) if return_gym_space else f(space)
