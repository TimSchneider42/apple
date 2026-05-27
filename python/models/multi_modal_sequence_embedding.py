from __future__ import annotations

from typing import Iterable, Protocol

import flax.linen as nn
import jax
import jax.numpy as jnp

from jax_util import tree_unfreeze
from pytree import PyTree
from .normalization import BaseNormalization
from .util import maybe_add_dims


class InputEncoder(Protocol):
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        pass


class NormalizationInputEncoder(nn.Module, InputEncoder):
    normalization: BaseNormalization

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        return self.normalization(
            x,
            evaluation_mode=evaluation_mode,
            mask=maybe_add_dims(
                norm_mask, x.ndim - (0 if norm_mask is None else norm_mask.ndim)
            ),
        )


class InputEncoderSequence(nn.Module, InputEncoder):
    sequence: Iterable[InputEncoder]

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        for encoder in self.sequence:
            x = encoder(x, evaluation_mode=evaluation_mode, norm_mask=norm_mask)
        return x


class MultiModalSequenceEmbedding(nn.Module):
    input_encoders: PyTree[InputEncoder]

    @nn.compact
    def __call__(
        self,
        input_sequences: PyTree[jax.Array],
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        embedding_tree = jax.tree.map(
            lambda encoder, sequence: encoder(
                sequence.reshape((-1,) + sequence.shape[2:]),
                evaluation_mode=evaluation_mode,
                norm_mask=None if norm_mask is None else norm_mask.reshape((-1,)),
            ).reshape(
                sequence.shape[:2] + (-1,),
            ),
            tree_unfreeze(self.input_encoders),
            tree_unfreeze(input_sequences),
        )

        return jnp.concatenate(jax.tree.flatten(embedding_tree)[0], axis=-1)
