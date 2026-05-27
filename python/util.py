from __future__ import annotations

import functools
import logging
import math
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Iterable, OrderedDict, Any, Callable, TypeVar

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def make_unique_dir(base_path: Path, suffix: str = "") -> Path:
    path = None
    done = False
    i = 0
    while not done:
        path = base_path.parent / "".join(
            [base_path.name, f"-{i}" if i != 0 else "", suffix]
        )
        try:
            path.mkdir(parents=True)
            done = True
        except OSError:
            if not path.exists():
                raise
        i += 1
    return path


def batched(iterable: Iterable, batch_size: int):
    if batch_size < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, batch_size)):
        yield batch


@dataclass
class TerminateSignal:
    set: bool = False


def fix_ordered_dicts(x: Any) -> Any:
    if isinstance(x, OrderedDict) or isinstance(x, dict):
        return {k: fix_ordered_dicts(v) for k, v in x.items()}
    elif isinstance(x, tuple):
        return tuple(fix_ordered_dicts(e) for e in x)
    else:
        return x


class LowFreqPrinter:
    def __init__(self, interval: int):
        self.__last_print: int | None = None
        self.__interval = interval

    def print(self, text: str, step: int):
        if self.__last_print is None or step - self.__last_print >= self.__interval:
            print(text)
            self.__last_print = step

    def __call__(self, text: str, step: int):
        self.print(text, step)


T = TypeVar("T")


def optional(inner_type: Callable[[str], T]) -> Callable[[str], T | None]:
    def inner(x: str) -> T | None:
        if x.lower() in ["none", "null", "nil", ""]:
            return None
        return inner_type(x)

    return inner


def closest_prime(n: int) -> int:
    primes = get_primes(max(2 * n, 1))
    primes = np.asarray(primes)
    dist = np.abs(primes - n)
    idx = np.argmin(dist)
    return int(primes[idx])


@functools.lru_cache(maxsize=1024)
def get_primes(n: int) -> tuple[int, ...]:
    # We count 1 as prime number here. Don't like it? Call the cops.
    if n < 2:
        return (1,)
    if n == 2:
        return 1, 2
    max_factor = int(math.ceil(math.sqrt(n)))
    prev_primes = get_primes(max_factor)
    remaining_numbers = np.arange(max_factor + 1, n + 1, dtype=np.int_)
    is_prime = np.all(
        remaining_numbers[:, None] % np.asarray(prev_primes)[None, 1:] != 0, axis=-1
    )
    return prev_primes + tuple(remaining_numbers[is_prime])


def get_max_episode_steps(vector_env: gym.vector.VectorEnv) -> int | None:
    if vector_env.spec is not None and vector_env.spec.max_episode_steps is not None:
        return vector_env.spec.max_episode_steps
    env_unwrapped = vector_env.unwrapped
    if isinstance(env_unwrapped, gym.vector.SyncVectorEnv):
        return max(
            (
                e
                for e in (get_max_episode_steps_single(e) for e in env_unwrapped.envs)
                if e is not None
            ),
            default=None,
        )
    elif isinstance(env_unwrapped, gym.vector.AsyncVectorEnv):
        dummy_envs = []
        try:
            for fn in env_unwrapped.env_fns:
                dummy_envs.append(fn())
            return max(
                (
                    e
                    for e in (get_max_episode_steps_single(e) for e in dummy_envs)
                    if e is not None
                ),
                default=None,
            )
        finally:
            for e in dummy_envs:
                e.close()
    return None


def get_max_episode_steps_single(env: gym.Env) -> int | None:
    current = env
    while current is not None:
        if current.spec is not None and current.spec.max_episode_steps is not None:
            return current.spec.max_episode_steps
        current = current.env if hasattr(current, "env") else None
    current = env
    while current is not None:
        for name in ["max_episode_steps", "max_episode_length"]:
            if hasattr(current, name):
                value = getattr(current, name)
                if isinstance(value, int):
                    logger.warning(
                        f"Unsafely extracted max episode steps from {type(current).__name__}.{name}: {value}"
                    )
                    return value
        current = current.env if hasattr(current, "env") else None
    return None
