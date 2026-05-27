from __future__ import annotations

import math
import typing
from abc import ABC, abstractmethod
from typing import Callable, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

from .attention import (
    AttentionMask,
    FlaxDotProductAttentionFn,
    AttentionFn,
)
from .multi_head_dot_product_attention import MultiHeadDotProductAttention
from .normalization import (
    LayerNorm,
    BaseNormalization,
)
from .util import maybe_add_dims


class TransformerBlock(nn.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        pass


class TransformerAttentionBlock(TransformerBlock, ABC):
    attention_fn: AttentionFn
    num_heads: int = 8

    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        pass

    @property
    def memory_alignment(self) -> int:
        return self.attention_fn.memory_alignment


class DefaultMLPBlock(TransformerBlock):
    dim_feedforward: int | None = None
    dropout_rate: float = 0.1
    activation: Callable[[jax.Array], jax.Array] = nn.activation.relu
    norm_before: bool = True
    norm_after: bool = False
    normalization: Callable[[], BaseNormalization] = LayerNorm

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        if self.norm_before:
            x_norm = self.normalization()(
                x,
                evaluation_mode=evaluation_mode,
                mask=maybe_add_dims(norm_mask, 1),
            )
        else:
            x_norm = x

        dim_feedforward = (
            x.shape[-1] if self.dim_feedforward is None else self.dim_feedforward
        )

        hidden1 = self.activation(nn.Dense(dim_feedforward, name="hidden1")(x_norm))
        hidden1_dropout = nn.Dropout(self.dropout_rate)(
            hidden1, deterministic=evaluation_mode
        )
        hidden2 = nn.Dense(x.shape[-1], name="hidden2")(hidden1_dropout)
        hidden2_dropout = nn.Dropout(self.dropout_rate)(
            hidden2, deterministic=evaluation_mode
        )

        output = x + hidden2_dropout

        if self.norm_after:
            return self.normalization()(
                output,
                evaluation_mode=evaluation_mode,
                mask=maybe_add_dims(norm_mask, 1),
            )
        else:
            return output


class DefaultMultiHeadSelfAttentionBlock(TransformerAttentionBlock):
    dropout_rate: float = 0.1
    norm_before: bool = True
    norm_after: bool = False
    normalization: Callable[[], BaseNormalization] = LayerNorm
    attention_fn: AttentionFn = FlaxDotProductAttentionFn(dropout_rate=0.1)
    apply_softmax_scaling: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        if self.norm_before:
            x_norm = self.normalization()(x, evaluation_mode=evaluation_mode)
        else:
            x_norm = x

        head_dim = x.shape[-1] // self.num_heads

        if self.apply_softmax_scaling:
            softmax_scaling_factor = 1 / (math.sqrt(head_dim))
        else:
            softmax_scaling_factor = 1

        multi_head_attention = MultiHeadDotProductAttention(
            qkv_features=x.shape[-1],
            out_features=x.shape[-1],
            num_heads=self.num_heads,
            attention_fn=self.attention_fn,
            softmax_scaling_factor=softmax_scaling_factor,
        )
        dropout1 = nn.Dropout(self.dropout_rate)
        hidden = dropout1(
            multi_head_attention(x_norm, mask=src_mask),
            deterministic=evaluation_mode,
        )

        output = x + hidden

        if self.norm_after:
            return self.normalization()(
                output,
                evaluation_mode=evaluation_mode,
                mask=maybe_add_dims(norm_mask, 1),
            )
        else:
            return output


class InputFn(typing.Protocol):
    def __call__(
        self,
        src: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array: ...


class LatentAggregationFn(typing.Protocol):
    def __call__(
        self,
        intermediate_outputs: tuple[jax.Array, ...],
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array: ...


class ConcatLatentAggregationFn(nn.Module, LatentAggregationFn):
    layer_weighting: Callable[[int], float] = lambda layer_idx: 1.0

    @nn.compact
    def __call__(
        self,
        intermediate_outputs: tuple[jax.Array, ...],
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        return jnp.concatenate(
            [self.layer_weighting(i) * c for i, c in enumerate(intermediate_outputs)],
            axis=-1,
        )


class ConcatFirstLastAggregationFn(nn.Module, LatentAggregationFn):
    first_weight: float = 1.0
    last_weight: float = 1.0

    @nn.compact
    def __call__(
        self,
        intermediate_outputs: tuple[jax.Array, ...],
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        if len(intermediate_outputs) < 2:
            return intermediate_outputs[0] * self.first_weight
        return jnp.concatenate(
            [
                intermediate_outputs[0] * self.first_weight,
                intermediate_outputs[1] * self.last_weight,
            ],
            axis=-1,
        )


class TransformerEncoder(nn.Module):
    layers: Sequence[TransformerBlock]
    input_fn: InputFn = lambda src, *args, **kwargs: src
    latent_aggregation_fn: LatentAggregationFn = (
        lambda intermediate_outputs, *args, **kwargs: intermediate_outputs[-1]
    )

    @nn.compact
    def __call__(
        self,
        src: jax.Array,
        src_mask: AttentionMask | None = None,
        src_pos: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        intermediate_outputs = [
            self.input_fn(
                src,
                src_mask=src_mask,
                src_pos=src_pos,
                evaluation_mode=evaluation_mode,
                norm_mask=norm_mask,
            )
        ]

        for layer in self.layers:
            latent = self.latent_aggregation_fn(
                tuple(intermediate_outputs),
                src_mask=src_mask,
                src_pos=src_pos,
                evaluation_mode=evaluation_mode,
                norm_mask=norm_mask,
            )
            intermediate_outputs.append(
                layer(
                    latent,
                    src_mask=src_mask,
                    src_pos=src_pos,
                    evaluation_mode=evaluation_mode,
                    norm_mask=norm_mask,
                )
            )

        return self.latent_aggregation_fn(
            tuple(intermediate_outputs),
            src_mask=src_mask,
            src_pos=src_pos,
            evaluation_mode=evaluation_mode,
            norm_mask=norm_mask,
        )

    @property
    def memory_alignment(self) -> int:
        return math.lcm(
            *(
                l.memory_alignment
                for l in self.layers
                if isinstance(l, TransformerAttentionBlock)
            ),
            1,
        )

    @property
    def num_heads(self):
        return math.lcm(
            *(
                l.num_heads
                for l in self.layers
                if isinstance(l, TransformerAttentionBlock)
            ),
            1,
        )
