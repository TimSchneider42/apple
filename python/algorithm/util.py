from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from data_logger import Loggable
from gym_space_map import gym_space_map
from pytree import PyTree


def mk_seed(rng: jax.Array = None) -> jax.Array:
    return jax.random.randint(rng, (), 0, jnp.iinfo(jnp.int32).max)


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for d in dicts:
        for k, v in d.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = merge_dicts(result[k], v)
            else:
                assert k not in result
                result[k] = v
    return result


def generate_log_dict(
    value: dict[str, Any] | jax.Array | np.ndarray | float | int,
    prefix: str | None = None,
    current_dict: dict[str, Loggable] | None = None,
) -> dict[str, Loggable]:
    if current_dict is None:
        current_dict = {}
    if isinstance(value, float) or isinstance(value, int):
        assert prefix is not None
        current_dict[prefix] = value
    elif isinstance(value, jax.Array) or isinstance(value, np.ndarray):
        if len(value.shape) == 0:
            assert prefix is not None
            current_dict[prefix] = value.item()
        else:
            prefix = "" if prefix is None else prefix + "/"
            for i in range(value.shape[0]):
                generate_log_dict(
                    value[i], prefix=f"{prefix}{i}", current_dict=current_dict
                )
    else:
        prefix = "" if prefix is None else prefix + "/"
        for k, v in value.items():
            if value.get(f"_{k}", True) and not k.startswith("_"):
                generate_log_dict(v, prefix=f"{prefix}{k}", current_dict=current_dict)
    return current_dict


def add_seq_dim(x: PyTree[jax.Array]) -> PyTree[jax.Array]:
    return jax.tree.map(lambda e: e[:, None], x)


def rem_seq_dim(x: PyTree[jax.Array]) -> PyTree[jax.Array]:
    return jax.tree.map(lambda e: e[:, 0], x)


def trajectory_variable_spec_from_space(
    space: gym.Space,
) -> PyTree[jax.ShapeDtypeStruct]:
    return gym_space_map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), space)
