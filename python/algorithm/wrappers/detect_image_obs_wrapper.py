from __future__ import annotations

import logging
from functools import partial

import gymnasium

from ap_gym import ImageSpace
from gym_space_map import gym_space_map

logger = logging.getLogger(__name__)


def _convert_space(
    space: gymnasium.Space, batch_shape: tuple[int, ...] = ()
) -> gymnasium.Space:
    if isinstance(space, gymnasium.spaces.Box) and not isinstance(space, ImageSpace):
        if len(space.shape) == 3 + len(batch_shape) and space.shape[-1] in [1, 3]:
            logger.warning(
                f"Auto-detected and converted image space with shape {space.shape}."
            )
            return ImageSpace(
                space.shape[-2],
                space.shape[-3],
                space.shape[-1],
                batch_shape=batch_shape,
                dtype=space.dtype,
                seed=space._np_random,
                low=space.low,
                high=space.high,
            )
    return space


class DetectImageObsWrapper(gymnasium.vector.VectorWrapper):
    def __init__(self, env: gymnasium.vector.VectorEnv):
        super().__init__(env)
        self._observation_space = gym_space_map(
            partial(_convert_space, batch_shape=(self.num_envs,)),
            env.observation_space,
            return_gym_space=True,
        )
        self.__obs_space_tree = gym_space_map(lambda x: x, self._observation_space)
        self._single_observation_space = gym_space_map(
            _convert_space, env.single_observation_space, return_gym_space=True
        )
