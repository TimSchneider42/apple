from __future__ import annotations

from typing import Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.image import ResizeMethod

from .multi_modal_sequence_embedding import InputEncoder


class ImageScaler(nn.Module, InputEncoder):
    target_image_shape: tuple[int, int]

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        if x.shape[-3:-1] == self.target_image_shape:
            return x
        return jax.image.resize(
            x,
            x.shape[:-3] + self.target_image_shape + (x.shape[-1],),
            method=ResizeMethod.LINEAR,
            antialias=True,
        )


class ChannelBroadcaster(nn.Module, InputEncoder):
    target_channel_count: int = 3

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        assert x.shape[-1] in [
            1,
            self.target_channel_count,
        ], f"Only 1 and {self.target_channel_count} channel images are supported."
        if x.shape[-1] == 1:
            x = jnp.broadcast_to(x, x.shape[:-1] + (self.target_channel_count,))
        return x


class ChannelAggregator(nn.Module, InputEncoder):
    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        return jnp.mean(x, axis=-1, keepdims=True)


class ImageConverter(nn.Module, InputEncoder):
    input_min: Union[float, int]
    input_max: Union[float, int]

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        image_float = x.astype(jnp.float32)
        return (image_float - self.input_min) / (self.input_max - self.input_min)
