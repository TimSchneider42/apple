from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import SupportsFloat, Any, Generic

import gymnasium as gym
import numpy as np
from ap_gym import ActivePerceptionVectorWrapper, ActivePerceptionActionSpace
from ap_gym.types import (
    ObsType,
    FullActType,
    ActType,
    PredType,
    ArrayType,
    PredTargetType,
)
from gymnasium.vector.utils import batch_space


class FixedControlStrategyWrapper(
    ActivePerceptionVectorWrapper[
        ObsType,
        tuple[()],
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
    ABC,
):
    def __init__(
        self,
        env: gym.vector.VectorEnv[ObsType, FullActType[ActType, PredType], ArrayType],
    ):
        super().__init__(env)
        self._single_action_space = ActivePerceptionActionSpace(
            inner_action_space=gym.spaces.Tuple(()),
            prediction_space=self.single_prediction_space,
        )
        self._action_space = batch_space(self._single_action_space, env.num_envs)
        self.__obs = self.__initial_step = self.__prev_done = None

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.__obs, *rest = super().reset(
            seed=seed,
            options=options,
        )
        self.__initial_step = np.ones((self.num_envs,), dtype=bool)
        self.__prev_done = np.zeros((self.num_envs,), dtype=bool)
        return self.__obs, *rest

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self.__obs, reward, terminated, truncated, info = super().step(
            {
                "action": self.generate_action(self.__obs, self.__initial_step),
                "prediction": action["prediction"],
            }
        )
        self.__initial_step = self.__prev_done
        self.__prev_done = terminated | truncated
        return self.__obs, reward, terminated, truncated, info

    @abstractmethod
    def generate_action(self, obs: ObsType, initial_step: np.ndarray) -> ActType:
        pass


class RandomActionWrapper(
    FixedControlStrategyWrapper[
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
    ):
        super().__init__(env)
        self.__inner_action_space = deepcopy(self.env.inner_action_space)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        rng = np.random.default_rng(seed)
        self.__inner_action_space.seed(int(rng.integers(0, 2**32 - 1)))
        return super().reset(
            seed=int(rng.integers(0, 2**32 - 1)),
            options=options,
        )

    def generate_action(self, obs: ObsType, initial_step: np.ndarray) -> ActType:
        return self.__inner_action_space.sample()


def identify_position_key(space: gym.spaces.Dict) -> str | None:
    candidates = [
        k
        for k, v in space.spaces.items()
        if isinstance(v, gym.spaces.Box)
        and v.shape in [(2,), (3,)]
        and "pos" in k.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        raise ValueError(
            f"Multiple position key candidates found in observation space: {candidates}"
        )
    else:
        raise ValueError("No position key candidate found in observation space.")


class FixedTrajectoryActionWrapper(
    FixedControlStrategyWrapper[
        dict[str, Any],
        np.ndarray,
        PredType,
        PredTargetType,
        np.ndarray,
    ],
    Generic[PredType, PredTargetType],
):
    def __init__(
        self,
        env: gym.vector.VectorEnv[
            ObsType, FullActType[np.ndarray, PredType], np.ndarray
        ],
    ):
        super().__init__(env)
        self.__position_key_obs = identify_position_key(self.single_observation_space)
        if isinstance(self.env.single_inner_action_space, gym.spaces.Dict):
            self.__position_key_act = identify_position_key(
                self.env.single_inner_action_space
            )
        elif isinstance(
            self.env.single_inner_action_space, gym.spaces.Box
        ) and self.env.single_inner_action_space.shape in [(2,), (3,)]:
            self.__position_key_act = None
        else:
            raise ValueError(
                "Inner action space must be either a Dict space or a Box space with shape (2,) or (3,)."
            )
        self.__t = None
        self.__reach_start_position_time_rel = None
        self.__initial_position = None
        self.__velocity_scale = None
        self.__traj_total_time = None

    def __estimate_velocity(self):
        velocities = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            obs, info = self.env.reset(seed=int(rng.integers(0, 2**32 - 1)))
            pos = obs[self.__position_key_obs]
            action_space = copy.deepcopy(self.env.action_space)
            action_space.seed(int(rng.integers(0, 2**32 - 1)))
            prev_done = np.zeros((self.num_envs,), dtype=bool)
            for i in range(max(self.spec.max_episode_steps, 100)):
                action = action_space.sample()
                inner_action = action["action"]
                if self.__position_key_act is not None:
                    vel_command = inner_action[self.__position_key_act]
                    inner_act_space = action_space.inner_action_space[
                        self.__position_key_act
                    ]
                else:
                    vel_command = inner_action
                    inner_act_space = action_space.inner_action_space
                choice = rng.integers(0, 2, size=vel_command.shape).astype(np.bool_)
                vel_command[:] = np.where(
                    choice, inner_act_space.low, inner_act_space.high
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                new_pos = obs[self.__position_key_obs]
                delta_pos = new_pos - pos
                velocities.extend((delta_pos / vel_command)[~prev_done])
                prev_done = terminated | truncated
                pos = new_pos
        self.__velocity_scale = np.median(np.linalg.norm(velocities, axis=-1), axis=0)

    @abstractmethod
    def _main_trajectory_normalized(
        self, t: np.ndarray | float, initial_position: np.ndarray
    ) -> np.ndarray:
        pass

    def __normalize_position(self, position: np.ndarray) -> np.ndarray:
        pos_space = self.observation_space.spaces[self.__position_key_obs]
        return (position - pos_space.low) / (pos_space.high - pos_space.low) * 2 - 1

    def __denormalize_length(self, length_norm: np.ndarray) -> np.ndarray:
        pos_space = self.observation_space.spaces[self.__position_key_obs]
        return length_norm / 2 * (pos_space.high - pos_space.low)

    def __denormalize_position(self, position_norm: np.ndarray) -> np.ndarray:
        pos_space = self.observation_space.spaces[self.__position_key_obs]
        return self.__denormalize_length(position_norm + 1) + pos_space.low

    def __trajectory(self, t: np.ndarray | float) -> np.ndarray:
        t_norm = t / self.__traj_total_time
        t_norm_adjusted_1 = t_norm / self.__reach_start_position_time_rel
        t_norm_adjusted_2 = (t_norm - self.__reach_start_position_time_rel) / (
            1 - self.__reach_start_position_time_rel
        )
        start_pos_normalized = self._main_trajectory_normalized(
            np.zeros(self.num_envs),
            self.__normalize_position(self.__initial_position)[..., :2],
        )
        if self.__initial_position.shape[-1] == 3:
            start_pos_normalized = np.concatenate(
                [start_pos_normalized, np.zeros((self.num_envs, 1))], axis=-1
            )
        start_pos = self.__denormalize_position(start_pos_normalized)
        pos_1 = (
            1 - t_norm_adjusted_1[:, None]
        ) * self.__initial_position + t_norm_adjusted_1[:, None] * start_pos
        pos_2_normalized = self._main_trajectory_normalized(
            t_norm_adjusted_2,
            self.__normalize_position(self.__initial_position)[..., :2],
        )
        if self.__initial_position.shape[-1] == 3:
            pos_2_normalized = np.concatenate(
                [pos_2_normalized, np.zeros((self.num_envs, 1))], axis=-1
            )
        pos_2 = self.__denormalize_position(pos_2_normalized)
        return np.where(
            (t_norm < self.__reach_start_position_time_rel)[:, None], pos_1, pos_2
        )

    def __reset_trajectories(
        self,
        obs: ObsType,
        where: np.ndarray | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if self.__initial_position is None:
            self.__initial_position = np.zeros_like(obs[self.__position_key_obs])
            self.__reach_start_position_time_rel = np.zeros(
                (self.num_envs,), dtype=np.float32
            )
            self.__traj_total_time = np.zeros((self.num_envs,), dtype=np.int_)
            self.__t = np.zeros((self.num_envs,), dtype=np.int_)
        if where is None:
            where = np.ones((self.num_envs,), dtype=np.bool_)
        if not np.any(where):
            return
        self.__initial_position[where] = obs[self.__position_key_obs][where]
        initial_pos = self.__initial_position
        start_pos_norm = self._main_trajectory_normalized(
            np.zeros(self.num_envs), self.__normalize_position(initial_pos)[..., :2]
        )
        if self.__initial_position.shape[-1] == 3:
            start_pos_norm = np.concatenate(
                [start_pos_norm, np.zeros((self.num_envs, 1))], axis=-1
            )
        start_pos = self.__denormalize_position(start_pos_norm)
        distance_to_start = np.linalg.norm(initial_pos - start_pos, axis=-1)
        positions = self._main_trajectory_normalized(
            np.linspace(0, 1, 1000)[:, None],
            self.__normalize_position(initial_pos)[..., :2],
        )
        main_traj_length = np.sum(
            np.linalg.norm(np.diff(positions, axis=0), axis=-1), axis=0
        )
        full_trajectory_length = main_traj_length + distance_to_start
        self.__reach_start_position_time_rel[where] = (
            distance_to_start / (full_trajectory_length)
        )[where]
        if self.__position_key_act is not None:
            act_space = self.env.single_inner_action_space[self.__position_key_act]
        else:
            act_space = self.env.single_inner_action_space
        max_vel = (
            np.linalg.norm(
                np.max(np.minimum(np.abs(act_space.low), np.abs(act_space.high)))
            )
            * self.__velocity_scale
        )
        self.__traj_total_time[where] = np.ceil(
            full_trajectory_length / max_vel
        ).astype(np.int_)[where]
        self.__t[where] = np.zeros((self.num_envs,), dtype=np.int32)[where]

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        if self.__velocity_scale is None:
            self.__estimate_velocity()
        obs, info = super().reset(seed=seed, options=options)
        self.__reset_trajectories(obs)
        return obs, info

    def generate_action(self, obs: ObsType, initial_step: np.ndarray) -> ActType:
        current_position = obs[self.__position_key_obs]
        self.__reset_trajectories(obs, where=initial_step)
        target_position = self.__trajectory(self.__t)
        self.__t += 1
        action = (target_position - current_position) / self.__velocity_scale
        if self.__position_key_act is None:
            return np.clip(
                action,
                self.env.inner_action_space.low,
                self.env.inner_action_space.high,
            )
        else:
            full_action = self.env.inner_action_space.sample()
            full_action[self.__position_key_act] = np.clip(
                action,
                self.env.inner_action_space[self.__position_key_act].low,
                self.env.inner_action_space[self.__position_key_act].high,
            )
            return full_action


class Grid2DActionWrapper(
    FixedTrajectoryActionWrapper[PredType, PredTargetType],
    Generic[PredType, PredTargetType],
):
    def _main_trajectory_normalized(
        self, t: np.ndarray | float, initial_position: np.ndarray
    ) -> np.ndarray:
        vertical_segments = 5
        horizontal_time = 1 / (vertical_segments + 1)
        vertical_time = 1 - horizontal_time
        horizontal_time_per_segment = horizontal_time / (vertical_segments - 1)
        vertical_time_per_segment = vertical_time / vertical_segments
        segment_times = np.where(
            np.arange(vertical_segments * 2 - 1) % 2 == 0,
            vertical_time_per_segment,
            horizontal_time_per_segment,
        )
        segment_end_times_cum = np.cumsum(segment_times)
        segment_start_times = np.concatenate(
            [np.array([0.0]), segment_end_times_cum[:-1]]
        )

        segment_idx = np.minimum(
            np.searchsorted(segment_end_times_cum, t, side="right"),
            len(segment_times) - 1,
        )

        # This generates a grid search pattern
        grid_search_pattern = np.stack(
            [
                np.repeat(np.linspace(-1, 1, vertical_segments, endpoint=True), 2),
                (((np.arange(vertical_segments * 2) + 1) // 2) % 2) * 2 - 1,
            ],
            axis=-1,
        )

        time_inside_segment = (t - segment_start_times[segment_idx]) / segment_times[
            segment_idx
        ]
        target_position = (
            grid_search_pattern[segment_idx] * (1 - time_inside_segment[..., None])
            + time_inside_segment[..., None] * grid_search_pattern[segment_idx + 1]
        )

        target_position = np.broadcast_to(
            target_position,
            target_position.shape[:-2] + (self.num_envs, target_position.shape[-1]),
        ).copy()

        # Flip the pattern such that the closest corner is visited first
        initial_corner = np.ceil(np.abs(initial_position)) * np.sign(initial_position)
        target_position[..., initial_corner[:, 0] == 1, 0] *= -1
        target_position[..., initial_corner[:, 1] == 1, 1] *= -1

        return target_position
