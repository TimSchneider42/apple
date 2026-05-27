from __future__ import annotations

import functools
from collections.abc import Callable
from functools import cached_property, lru_cache, partial
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp


def combine_masks(*masks: jax.Array | None) -> jax.Array | None:
    masks_list = [m for m in masks if m is not None]
    if not masks_list:
        return None
    mask, *other_masks = masks_list
    for other_mask in other_masks:
        mask = jnp.logical_and(mask, other_mask)
    return mask


def cached_class_method(maxsize: int | None = 128) -> Callable[..., Any]:
    def outer(func: Callable[..., Any]):
        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            cache_fn_name = f"_cache_{func.__name__}"
            if cache_fn_name not in self.__dict__:
                self.__dict__[cache_fn_name] = lru_cache(maxsize=maxsize)(
                    partial(func, self)
                )
            return self.__dict__[cache_fn_name](*args, **kwargs)

        return inner

    return outer


@flax.struct.dataclass
class AttentionMask:
    episode_ids: jax.Array | None = None
    past_horizon: int | None = None
    future_horizon: int | None = None
    flat_mask: jax.Array | None = None
    base_mask: jax.Array | None = None

    @cached_property
    def is_causal(self) -> bool:
        return self.future_horizon == 0

    @cached_property
    def same_episode_mask(self) -> jax.Array | None:
        if self.episode_ids is None:
            return None
        return self.episode_ids[:, :, None] == self.episode_ids[:, None]

    @cached_class_method(maxsize=None)
    def __get_step_idx(self, sequence_length: int | None = None) -> jax.Array | None:
        return jnp.arange(self.__get_sequence_length(sequence_length))

    @cached_class_method(maxsize=None)
    def get_future_horizon_mask(
        self, sequence_length: int | None = None
    ) -> jax.Array | None:
        if self.future_horizon is None or self.future_horizon >= sequence_length - 1:
            return None
        step_idx = self.__get_step_idx(sequence_length)
        return (step_idx[None] - step_idx[:, None] <= self.future_horizon)[None]

    @cached_class_method(maxsize=None)
    def get_past_horizon_mask(
        self, sequence_length: int | None = None
    ) -> jax.Array | None:
        sequence_length = self.__get_sequence_length(sequence_length)
        if self.past_horizon is None or self.past_horizon >= sequence_length - 1:
            return None
        step_idx = self.__get_step_idx(sequence_length)
        return (step_idx[:, None] - step_idx[None] <= self.past_horizon)[None]

    @cached_class_method(maxsize=None)
    def get_horizon_mask(self, sequence_length: int | None = None) -> jax.Array | None:
        return combine_masks(
            self.get_past_horizon_mask(sequence_length),
            self.get_future_horizon_mask(sequence_length),
        )

    @cached_property
    def horizon_mask(self) -> jax.Array | None:
        return self.get_horizon_mask()

    @cached_class_method(maxsize=None)
    def get_full_mask(
        self,
        sequence_length: int | None = None,
        include_same_episode_mask: bool = True,
        include_past_horizon_mask: bool = True,
        include_future_horizon_mask: bool = True,
        include_flat_mask: bool = True,
        include_base_mask: bool = True,
        return_as_float_mask: bool = False,
    ) -> jax.Array | None:
        bool_mask = combine_masks(
            (
                self.get_past_horizon_mask(sequence_length)
                if include_past_horizon_mask
                else None
            ),
            (
                self.get_future_horizon_mask(sequence_length)
                if include_future_horizon_mask
                else None
            ),
            self.same_episode_mask if include_same_episode_mask else None,
            self.flat_mask_expanded if include_flat_mask else None,
            self.base_mask if include_base_mask else None,
        )
        if return_as_float_mask and bool_mask is not None:
            return jnp.where(bool_mask, 0, -jnp.inf)
        return bool_mask

    @cached_property
    def full_mask(self) -> jax.Array | None:
        return self.get_full_mask()

    @cached_property
    def flat_mask_expanded(self):
        if self.flat_mask is None:
            return None
        return self.flat_mask[..., None, :] | jnp.eye(
            self.flat_mask.shape[-1], dtype=jnp.bool_
        )

    @cached_property
    def sequence_length(self) -> int | None:
        sequence_length = None
        if self.episode_ids is not None:
            sequence_length = self.episode_ids.shape[-1]
        if self.flat_mask is not None:
            if (
                sequence_length is not None
                and sequence_length != self.flat_mask.shape[-1]
            ):
                raise ValueError("Inconsistent sequence lengths in attention mask.")
            else:
                sequence_length = self.flat_mask.shape[-1]
        if self.base_mask is not None:
            if self.base_mask.shape[-2] != self.base_mask.shape[-1]:
                raise ValueError("Base mask must have shape [..., L, L] for some L.")
            if (
                sequence_length is not None
                and sequence_length != self.base_mask.shape[-1]
            ):
                raise ValueError("Inconsistent sequence lengths in attention mask.")
            else:
                sequence_length = self.base_mask.shape[-1]
        return sequence_length

    def __get_sequence_length(self, sequence_length: int | None = None) -> int:
        if sequence_length is None:
            if self.sequence_length is None:
                raise ValueError(
                    "Either sequence_length must be provided or episode_ids/flat_mask/base_mask must be set."
                )
            return self.sequence_length
        else:
            if (
                self.sequence_length is not None
                and sequence_length != self.sequence_length
            ):
                raise ValueError(
                    "Inconsistent sequence lengths between provided sequence_length and "
                    "episode_ids/flat_mask/base_mask."
                )
            return sequence_length

    def __and__(self, other: jax.Array):
        if not isinstance(other, jax.Array):
            return NotImplemented
        return self.replace(base_mask=combine_masks(self.base_mask, other))
