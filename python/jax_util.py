from __future__ import annotations

from functools import partial
from typing import Mapping, Union, TypeVar, Generic, Any, Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from jax import custom_vjp

from pytree import PyTree


def tree_unfreeze(x: PyTree[jax.Array]) -> PyTree[jax.Array]:
    return jax.tree.map(
        lambda e: tree_unfreeze(e.unfreeze()) if isinstance(e, FrozenDict) else e,
        x,
        is_leaf=lambda e: isinstance(e, FrozenDict),
    )


def tree_freeze(x: PyTree[jax.Array]) -> PyTree[jax.Array]:
    return jax.tree.map(
        lambda e: tree_freeze(FrozenDict(e)) if isinstance(e, dict) else e,
        x,
        is_leaf=lambda e: isinstance(e, dict),
    )


def tree_sum(value: PyTree[jax.Array], batch_shape: tuple[int, ...]) -> jax.Array:
    return jax.tree.reduce(
        lambda c, y: c + jnp.sum(y, axis=tuple(range(len(batch_shape), len(y.shape)))),
        value,
        initializer=jnp.zeros(batch_shape),
    )


T = TypeVar("T")


class RecursiveDict(dict[str, Union["RecursiveDict[T]", T]], Generic[T]):
    pass


class RecursiveMapping(Mapping[str, Union["RecursiveMapping[T]", T]], Generic[T]):
    pass


def dict_mask(
    variables: RecursiveMapping[T], mask: RecursiveMapping[bool]
) -> RecursiveDict[T]:
    return {
        k: dict_mask(v, mask[k]) if isinstance(v, Mapping) else v
        for k, v in variables.items()
        if mask.get(k)
    }


def _dict_update_single(
    key: str, variables: Union[RecursiveMapping[T], T], updates: RecursiveMapping[T]
) -> Union[RecursiveMapping[T], T]:
    if key not in updates:
        return variables
    update = updates[key]
    if isinstance(variables, Mapping):
        return dict_update(variables, update)
    else:
        return update


def dict_update(
    variables: RecursiveMapping[T], updates: RecursiveMapping[T]
) -> RecursiveMapping[T]:
    return {k: _dict_update_single(k, v, updates) for k, v in variables.items()}


def get_act_fn(act_fn: str):
    return {
        "relu": nn.relu,
        "leaky_relu": nn.leaky_relu,
        "tanh": nn.tanh,
        "sinus": jnp.sin,
    }[act_fn]


def _path_to_str(path: list[jax.tree.PathEntry], root: bool = True) -> str:
    if len(path) == 0:
        return ""
    element, rest = path[0], path[1:]
    if isinstance(element, jax.tree_util.DictKey):
        if root:
            output = element.key
        else:
            output = "." + element.key
    elif isinstance(element, jax.tree_util.SequenceKey):
        if root:
            raise ValueError("Root path element cannot be a SequenceKey")
        output = f"[{element.idx}]"
    elif isinstance(element, jax.tree_util.GetAttrKey):
        if root:
            raise ValueError("Root path element cannot be a GetAttrKey")
        output = f"[{element.name}]"
    else:
        raise ValueError(f"Unsupported path element type: {type(element)}")
    return output + _path_to_str(rest, root=False)


def path_to_str(path: list[jax.tree.PathEntry]) -> str:
    return _path_to_str(path)


def compute_param_metrics(
    name: str, value: PyTree[jax.Array | float]
) -> dict[str, Any]:
    value_flat = {path_to_str(p): e for p, e in jax.tree.leaves_with_path(value)}
    if len(value_flat) == 0:
        return {}
    value_lst = list(value_flat.values())
    if any(isinstance(e, jax.Array) for e in value_lst):
        value_arr = jnp.array(value_lst)
    else:
        value_arr = np.array(value_lst)
    return {
        name: {
            "min": value_arr.min(),
            "max": value_arr.max(),
            "mean": value_arr.mean(),
        },
        f"{name}_individual": value_flat,
    }


def metrics_set_valid_flag(
    metrics: dict[str, Any] | None,
    override_empty: bool = False,
    zeros_like_fn: Callable[[jax.Array], jax.Array | np.ndarray] = jnp.zeros_like,
) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        **{
            k: metrics_set_valid_flag(v, override_empty, zeros_like_fn=zeros_like_fn)
            for k, v in metrics.items()
            if isinstance(v, dict)
        },
        **{
            f"_{k}": not override_empty and metrics.get(f"_{k}", True)
            for k, v in metrics.items()
            if not isinstance(v, dict) and not k.startswith("_")
        },
        **{
            k: zeros_like_fn(v) if override_empty else v
            for k, v in metrics.items()
            if not isinstance(v, dict)
        },
    }
