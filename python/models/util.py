from __future__ import annotations

from typing import Mapping, Iterable, overload

import flax.typing
import jax
from flax.core import FrozenDict


def update_existing_recursive(target_dict: Mapping, update_dict: Mapping) -> FrozenDict:
    return FrozenDict(
        {
            k: (
                update_existing_recursive(target_dict[k], update_dict[k])
                if isinstance(update_dict[k], Mapping)
                else update_dict[k]
            )
            for k in target_dict
        }
    )


def canonicalize_axes(rank: int, axes: flax.typing.Axes) -> tuple[int, ...]:
    if not isinstance(axes, Iterable):
        axes = (axes,)
    return tuple({rank + axis if axis < 0 else axis for axis in axes})


@overload
def maybe_add_dims(x: None, additional_dims: int) -> None: ...


@overload
def maybe_add_dims(x: jax.Array, additional_dims: int) -> jax.Array: ...


def maybe_add_dims(x: jax.Array | None, additional_dims: int) -> jax.Array | None:
    if x is None:
        return None
    return x.reshape(x.shape + (1,) * additional_dims)
