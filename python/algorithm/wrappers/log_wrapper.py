from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from enum import Enum
from typing import SupportsFloat, Any, Iterable, Sequence

import gymnasium as gym
import jax
import numpy as np

from ap_gym import (
    ActivePerceptionVectorWrapper,
    BaseActivePerceptionVectorEnv,
)
from data_logger import Video, Loggable, BaseDataLogger
from exponential_moving_average import ExponentialMovingAverage
from gym_space_map import gym_space_map
from rate import Rate
from util import LowFreqPrinter

logger = logging.getLogger(__name__)


class RecorderState(Enum):
    INACTIVE = 0
    READY = 1
    RECORDING = 2
    DONE = 3


class VideoRecorder:
    def __init__(self, fps: int, rate: Rate):
        self.__state = RecorderState.INACTIVE
        self.__current_sequence = deque()
        self.__fps = fps
        self.__rate = rate

    def finalize(self) -> Video | None:
        if self.__state in [RecorderState.RECORDING, RecorderState.DONE]:
            output = Video(np.stack(self.__current_sequence, axis=0), self.__fps)
            self.__rate.do()
            self.clear()
            return output
        self.clear()
        return None

    def record(self, frame: np.ndarray, done: bool = False):
        if self.__state == RecorderState.READY:
            self.__state = RecorderState.RECORDING
        if self.__state == RecorderState.RECORDING:
            self.__current_sequence.append(frame)
            if done:
                self.__state = RecorderState.DONE

    def activate(self):
        if self.__state == RecorderState.INACTIVE:
            self.__state = RecorderState.READY

    def clear(self):
        self.__state = RecorderState.INACTIVE
        self.__current_sequence.clear()

    @property
    def state(self) -> RecorderState:
        return self.__state

    @property
    def empty(self):
        return len(self) == 0

    @property
    def active(self):
        return self.__state != RecorderState.INACTIVE

    @property
    def done(self):
        return self.__state == RecorderState.DONE

    @property
    def rate(self) -> Rate:
        return self.__rate

    def __len__(self):
        return len(self.__current_sequence)


class LogWrapper(ActivePerceptionVectorWrapper):
    def __init__(
        self,
        env: BaseActivePerceptionVectorEnv,
        data_logger: BaseDataLogger,
        prefix: str = "env",
        print_interval: int | None = None,
        smoothing_factor: float = 0.9,
        video_log_interval: int | None = None,
        stat_log_interval: int | None = None,
        return_log_interval: int | None = None,
    ):
        super().__init__(env)
        self.data_logger = data_logger
        self.total_environment_steps = 0
        self.__internal_step = 0
        self.prefix = prefix
        self.__printer = (
            None if print_interval is None else LowFreqPrinter(print_interval)
        )
        self.__return_filter = ExponentialMovingAverage(smoothing_factor)
        self.__video_log_interval = video_log_interval
        self.__episode_is_empty = True
        self.__prev_step_time = None
        fps = self.metadata.get("render_fps", 10)
        self.__video_recorders = [
            VideoRecorder(
                fps,
                (
                    Rate(video_log_interval)
                    if video_log_interval is not None
                    else Rate.never()
                ),
            )
            for _ in range(self.num_envs)
        ]
        self.__stat_log_rates = {
            k: defaultdict(
                lambda: (
                    Rate(stat_log_interval)
                    if stat_log_interval is not None
                    else Rate.never()
                )
            )
            for k in ["scalar", "vector"]
        }
        self.__return_log_rate = (
            Rate(return_log_interval)
            if return_log_interval is not None
            else Rate.never()
        )
        self.__timing_log_rate = (
            Rate(stat_log_interval) if stat_log_interval is not None else Rate.never()
        )
        if self.render_mode != "rgb_array":
            logger.warning(
                f"Not logging videos for environment {prefix} as render mode is {self.render_mode} and not "
                f"'rgb_array'."
            )
            self.__video_log_interval = None
        self.__tracked_stats = {
            "episodic_length": np.zeros(self.num_envs, dtype=np.int_),
            "episodic_return": np.zeros(self.num_envs, dtype=np.float32),
            "episodic_base_return": np.zeros(self.num_envs, dtype=np.float32),
            "episodic_prediction_loss": np.zeros(self.num_envs, dtype=np.float32),
            "episodic_abs_real_time": np.zeros(self.num_envs, dtype=np.float32),
            "episodic_own_real_time": np.zeros(self.num_envs, dtype=np.float32),
            **{
                "_episodic_action_mag_sum/"
                + ".".join(e.key for e in p): np.zeros(self.num_envs, dtype=np.float32)
                for p, s in jax.tree.flatten_with_path(
                    gym_space_map(lambda x: x, self.action_space)
                )[0]
                if isinstance(s, gym.spaces.Box)
            },
        }
        self.__prev_done = np.zeros(self.num_envs, dtype=bool)
        self.__max_return = -np.inf

    def __finalize_video_logs(self, indices: Iterable[int]) -> dict[str, Video]:
        return {
            f"{self.prefix}/video_{i}": self.__video_recorders[i].finalize()
            for i in indices
            if not self.__video_recorders[i].empty
        }

    def __maybe_activate_recorders(self, mask: np.ndarray):
        if self.__video_log_interval is not None:
            for rec, m in zip(self.__video_recorders, mask):
                if m and rec.rate.due(self.__internal_step):
                    rec.activate()

    def __render_and_return_complete(self, done: np.ndarray):
        if self.__video_log_interval is None:
            return {}

        if any(rec.active for rec in self.__video_recorders):
            rendered_images = self.render()
            for rec, img, d in zip(self.__video_recorders, rendered_images, done):
                rec.record(img, d)

        return self.__finalize_video_logs(
            i for i, rec in enumerate(self.__video_recorders) if rec.done
        )

    @staticmethod
    def __ragged_vector_mean(data: Sequence[Sequence[float]]) -> np.ndarray:
        max_length = max(len(d) for d in data)
        padded = np.stack([np.pad(d, (0, max_length - len(d))) for d in data], axis=0)
        return np.mean(padded, axis=0)

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        step_start_time = time.time()
        obs, reward, terminated, truncated, info = super().step(action)
        step_end_time = time.time()
        self.__internal_step += self.num_envs
        self.total_environment_steps += self.num_envs
        stat_updates = {
            "episodic_length": np.ones(self.num_envs, dtype=np.int_),
            "episodic_return": reward,
            "episodic_base_return": info.get("base_reward", np.zeros(self.num_envs)),
            "episodic_prediction_loss": info.get("prediction", {}).get(
                "loss", np.zeros(self.num_envs)
            ),
            "episodic_abs_real_time": np.full(
                self.num_envs, step_end_time - self.__prev_step_time
            ),
            "episodic_own_real_time": np.full(
                self.num_envs, step_end_time - step_start_time
            ),
            **{
                "_episodic_action_mag_sum/"
                + ".".join(e.key for e in p): np.linalg.norm(a, axis=-1)
                for p, a in jax.tree.flatten_with_path(
                    jax.tree.map(
                        lambda s, a: a if isinstance(s, gym.spaces.Box) else None,
                        gym_space_map(lambda x: x, self.action_space),
                        action,
                    )
                )[0]
                if a is not None
            },
        }
        for k in self.__tracked_stats:
            self.__tracked_stats[k][self.__prev_done] = 0
            self.__tracked_stats[k][~self.__prev_done] += stat_updates[k][
                ~self.__prev_done
            ]
        self.__prev_done = done = terminated | truncated

        stats = defaultdict(lambda: [])
        stats_vec = defaultdict(lambda: [])
        if "stats" in info:
            for name, target in [("scalar", stats), ("vector", stats_vec)]:
                if name in info["stats"]:
                    for k, v in info["stats"][name].items():
                        if not k.startswith("_") and self.__stat_log_rates[name][k](
                            self.__internal_step
                        ):
                            target[k].extend(
                                np.asarray(v)[
                                    info["stats"][name].get(
                                        f"_{k}", np.ones(self.num_envs, dtype=bool)
                                    )
                                ]
                            )

        max_ret = self.__max_return
        if np.any(done):
            if self.__return_log_rate(self.__internal_step):
                stats.update(
                    {
                        k: v[done]
                        for k, v in self.__tracked_stats.items()
                        if not k.startswith("_")
                    }
                )
                stats.update(
                    {
                        f"episodic_action_mag/{k.split('/')[1]}": v[done]
                        / self.__tracked_stats["episodic_length"][done]
                        for k, v in self.__tracked_stats.items()
                        if k.startswith("_episodic_action_mag_sum")
                    }
                )
                stats["episodic_mean_reward"] = (
                    self.__tracked_stats["episodic_return"][done]
                    / self.__tracked_stats["episodic_length"][done]
                )
                stats["episodic_mean_prediction_loss"] = (
                    self.__tracked_stats["episodic_prediction_loss"][done]
                    / self.__tracked_stats["episodic_length"][done]
                )
            if self.__timing_log_rate(self.__internal_step):
                num_steps = self.__tracked_stats["episodic_length"][done]
                abs_time = self.__tracked_stats["episodic_abs_real_time"][done]
                own_time = self.__tracked_stats["episodic_own_real_time"][done]
                stats.update(
                    {
                        "effective_step_freq": num_steps / abs_time,
                        "step_freq": num_steps / own_time,
                    }
                )
            ret = self.__return_filter(
                np.mean(self.__tracked_stats["episodic_return"][done])
            )
            max_ret = np.max(self.__tracked_stats["episodic_return"][done])
            self.__printer(
                f"[{self.total_environment_steps: 8d}] {self.prefix} episodic return: {ret:0.2f}",
                self.__internal_step,
            )

        mean_stats = {k: float(np.mean(v)) for k, v in stats.items() if len(v) > 0}
        mean_stats_vec = {
            k: self.__ragged_vector_mean(v) for k, v in stats_vec.items() if len(v) > 0
        }

        log_dict: dict[str, Loggable] = {
            **{f"{self.prefix}/{k}": v for k, v in mean_stats.items()},
            **{f"{self.prefix}_vec/{k}": v for k, v in mean_stats_vec.items()},
        }
        if max_ret > self.__max_return:
            self.__max_return = max_ret
            log_dict[f"{self.prefix}/max_episodic_return"] = self.__max_return
        log_dict.update(self.__render_and_return_complete(done))
        self.__maybe_activate_recorders(done)
        if len(log_dict) > 0:
            self.data_logger.write(log_dict, self.total_environment_steps)
        self.__episode_is_empty = False
        self.__prev_step_time = step_end_time
        return obs, reward, terminated, truncated, info

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        if self.__video_log_interval is not None:
            if not self.__episode_is_empty:
                self.data_logger.write(
                    self.__finalize_video_logs(range(self.num_envs)),
                    self.total_environment_steps,
                )
            else:
                # Reset was called twice in a row without any steps in between, and we were logging videos.
                # In this case, keep logging videos but start a new sequence.
                for rec in self.__video_recorders:
                    if rec.active:
                        rec.clear()
                        rec.activate()
        self.__episode_is_empty = True
        self.__maybe_activate_recorders(np.ones(self.num_envs, dtype=bool))
        self.__render_and_return_complete(np.zeros(self.num_envs, dtype=bool))
        self.__prev_done.fill(0)
        for v in self.__tracked_stats.values():
            v.fill(0)
        self.__prev_step_time = time.time()
        return obs, info
