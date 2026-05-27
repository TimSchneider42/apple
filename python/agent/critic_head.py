from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from bounding_functions import BoundingFunction
from models import (
    MultiModalSequenceEmbedding,
    BaseNormalization,
    NoNorm,
    TransformerBlock,
    CompositeBoundedDenseGeneralFactory,
    BOUNDED_DENSE_GENERAL_FACTORIES,
)
from models.util import maybe_add_dims
from pytree import PyTree


class CriticHead(nn.Module, ABC):
    @abstractmethod
    @nn.compact
    def __call__(
        self, hidden_state: jax.Array, *, evaluation_mode: bool = False
    ) -> jax.Array:
        pass


class ValueFunctionCriticHead(CriticHead):
    kernel_init: nn.initializers.Initializer = nn.linear.default_kernel_init

    @nn.compact
    def __call__(
        self, hidden_state: jax.Array, evaluation_mode: bool = False
    ) -> jax.Array:
        return nn.Dense(features=1, kernel_init=self.kernel_init)(hidden_state)[..., 0]


class QFunctionBackbone(ABC):
    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        pass


class DenseQFunctionBackbone(nn.Module, QFunctionBackbone):
    hidden_dims: int | None = 64
    hidden_layers: int = 2
    normalization: Callable[[], BaseNormalization] = NoNorm
    act_fn: Callable[[jax.Array], jax.Array] = nn.relu
    skip_connections: bool = False

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        hidden_dims = x.shape[-1] if self.hidden_dims is None else self.hidden_dims

        if self.skip_connections:
            prev_skip = x
            if prev_skip.shape[-1] < hidden_dims:
                pad_width = (((0, 0),) * (len(prev_skip.shape) - 1)) + (
                    (0, hidden_dims - prev_skip.shape[-1]),
                )
                prev_skip = jnp.pad(
                    prev_skip, pad_width=pad_width, mode="constant", constant_values=0
                )
            if prev_skip.shape[-1] > hidden_dims:
                prev_skip = prev_skip[..., :hidden_dims]
        else:
            prev_skip = None

        intermediate_state = self.normalization()(
            x,
            evaluation_mode=evaluation_mode,
            mask=maybe_add_dims(norm_mask, 1),
        )
        for i in range(self.hidden_layers):
            intermediate_state = nn.Dense(
                features=hidden_dims,
                name=f"intermediate_layer_{i}",
            )(
                intermediate_state,
            )
            intermediate_state = self.normalization()(intermediate_state)
            if self.skip_connections and (i + 1) % 2 == 0:
                intermediate_state += prev_skip
                prev_skip = intermediate_state
            intermediate_state = self.act_fn(intermediate_state)
        return intermediate_state


class QFunctionCriticHead(CriticHead):
    action_embedding: MultiModalSequenceEmbedding
    action_hint_embedding_dims: int | None = None
    state_embedding_dims: int | None = None
    backbone: QFunctionBackbone = DenseQFunctionBackbone()
    output_shape: tuple[int, ...] = ()
    action_hint_normalization: Callable[[], BaseNormalization] = NoNorm
    state_normalization: Callable[[], BaseNormalization] = NoNorm
    post_encoding_norm: Callable[[], BaseNormalization] = NoNorm
    hint_embedding: MultiModalSequenceEmbedding | None = None

    @nn.compact
    def __call__(
        self,
        hidden_state: jax.Array,
        action: PyTree[jax.Array],
        hint: PyTree[jax.Array] = None,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        hidden_state = self.state_normalization()(
            hidden_state,
            evaluation_mode=evaluation_mode,
            mask=maybe_add_dims(norm_mask, 1),
        )
        if self.state_embedding_dims is not None:
            hidden_state_encoded = nn.Dense(
                features=self.state_embedding_dims,
                name=f"hidden_state_encoding",
            )(
                hidden_state,
                evaluation_mode=evaluation_mode,
                norm_mask=maybe_add_dims(norm_mask, 1),
            )
            hidden_state_encoded = self.post_encoding_norm()(hidden_state_encoded)
        else:
            hidden_state_encoded = hidden_state

        if len(jax.tree.flatten(action)[0]) > 0:
            action_embedded = [
                self.action_embedding(
                    action, evaluation_mode=evaluation_mode, norm_mask=norm_mask
                )
            ]
        else:
            action_embedded = []
        if self.hint_embedding is not None:
            hint_embedded = [
                self.hint_embedding(
                    hint, evaluation_mode=evaluation_mode, norm_mask=norm_mask
                )
            ]
        else:
            hint_embedded = []
        inputs_embedded_lst = action_embedded + hint_embedded
        if len(inputs_embedded_lst) > 0:
            inputs_embedded = self.action_hint_normalization()(
                jnp.concatenate(inputs_embedded_lst, axis=-1),
                evaluation_mode=evaluation_mode,
                mask=maybe_add_dims(norm_mask, 1),
            )
            if self.action_hint_embedding_dims is not None:
                inputs_encoded = nn.Dense(
                    features=self.action_hint_embedding_dims,
                    name=f"action_hint_encoding",
                )(
                    inputs_embedded,
                    evaluation_mode=evaluation_mode,
                    norm_mask=maybe_add_dims(norm_mask, 1),
                )
                inputs_encoded = self.post_encoding_norm()(inputs_encoded)
            else:
                inputs_encoded = inputs_embedded
            intermediate_state = jnp.concatenate(
                [hidden_state_encoded, inputs_encoded], axis=-1
            )
        else:
            intermediate_state = hidden_state_encoded

        intermediate_state = self.post_encoding_norm()(
            intermediate_state,
            evaluation_mode=evaluation_mode,
            mask=maybe_add_dims(norm_mask, 1),
        )

        intermediate_state = self.backbone(
            intermediate_state, evaluation_mode=evaluation_mode, norm_mask=norm_mask
        )

        if self.output_shape == ():
            return nn.Dense(1)(intermediate_state)[..., 0]
        else:
            return nn.DenseGeneral(self.output_shape)(intermediate_state)


class QFunctionCriticEnsembleHead(CriticHead):
    action_embedding: MultiModalSequenceEmbedding
    action_hint_embedding_dims: int | None = None
    state_embedding_dims: int | None = None
    backbone: QFunctionBackbone = DenseQFunctionBackbone()
    output_shape: tuple[int, ...] = ()
    action_hint_normalization: Callable[[], BaseNormalization] = NoNorm
    state_normalization: Callable[[], BaseNormalization] = NoNorm
    post_encoding_norm: Callable[[], BaseNormalization] = NoNorm
    hint_embedding: MultiModalSequenceEmbedding | None = None
    ensemble_size: int = 1

    @nn.compact
    def __call__(
        self,
        hidden_state: jax.Array,
        action: PyTree[jax.Array],
        hint: PyTree[jax.Array] = None,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        q_function_vmap = nn.vmap(
            QFunctionCriticHead,
            variable_axes={True: 0},  # variables not shared between the critics
            split_rngs={True: True},  # different rngs
            in_axes=None,
            out_axes=0,
            axis_size=self.ensemble_size,
        )
        fn = q_function_vmap(
            action_embedding=self.action_embedding,
            action_hint_embedding_dims=self.action_hint_embedding_dims,
            state_embedding_dims=self.state_embedding_dims,
            backbone=self.backbone,
            action_hint_normalization=self.action_hint_normalization,
            state_normalization=self.state_normalization,
            post_encoding_norm=self.post_encoding_norm,
            output_shape=self.output_shape,
            hint_embedding=self.hint_embedding,
        )
        return fn(hidden_state, action, hint, evaluation_mode, norm_mask)
