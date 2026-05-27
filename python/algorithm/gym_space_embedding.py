from __future__ import annotations

from functools import partial
from typing import Any, Mapping, Type, Dict, Protocol, Callable

import flax.typing
import gymnasium
import jax.nn

from gym_space_map import gym_space_map
from models import (
    InputEncoderSequence,
    NormalizationInputEncoder,
    BaseNormalization,
    InputEncoder,
)
from .wrappers import Discrete32


class EmbeddingFactory(Protocol):
    def __call__(self, space: gymnasium.Space) -> InputEncoder: ...


def _flatten(x: jax.Array, flatten_dims: int) -> jax.Array:
    if flatten_dims == 0:
        return x
    return x.reshape(x.shape[: len(x.shape) - flatten_dims] + (-1,))


class InputNormalizationFactory(Protocol):
    def __call__(self, *, axis: flax.typing.Axes) -> BaseNormalization: ...


def mk_discrete32_factory(
    input_normalization: InputNormalizationFactory | None = None,
) -> EmbeddingFactory:
    def factory(space: Discrete32) -> InputEncoder:
        def embedding_fn(
            x: jax.Array, *, evaluation_mode: bool, norm_mask: jax.Array
        ) -> jax.Array:
            return jax.nn.one_hot(x, num_classes=space.n)

        if input_normalization is not None:
            embedding_fn = InputEncoderSequence(
                [embedding_fn, NormalizationInputEncoder(input_normalization(axis=-1))]
            )

        return embedding_fn

    return factory


def mk_box_factory(
    input_normalization: InputNormalizationFactory | None = None,
) -> EmbeddingFactory:
    def factory(space: gymnasium.spaces.Box) -> InputEncoder:
        def embedding_fn(
            x: jax.Array, *, evaluation_mode: bool, norm_mask: jax.Array
        ) -> jax.Array:
            return _flatten(x, flatten_dims=len(space.shape))

        if input_normalization is not None:
            embedding_fn = InputEncoderSequence(
                [
                    embedding_fn,
                    NormalizationInputEncoder(
                        input_normalization(axis=range(-len(space.shape), 0))
                    ),
                ]
            )

        return embedding_fn

    return factory


def get_default_embedding_factories(
    input_normalization: InputNormalizationFactory | None = None,
) -> Dict[Type[gymnasium.Space], EmbeddingFactory]:
    return {
        Discrete32: mk_discrete32_factory(input_normalization=input_normalization),
        gymnasium.spaces.Box: mk_box_factory(input_normalization=input_normalization),
    }


def _mk_embedding_fn(
    space: gymnasium.Space,
    embedding_factories: Mapping[Type[gymnasium.Space], EmbeddingFactory],
):
    supertypes = type(space).__mro__
    for supertype in supertypes:
        if supertype in embedding_factories:
            return embedding_factories[supertype](space)
    raise NotImplementedError(f"Unsupported space {space}")


def embeddings_from_space(
    space: gymnasium.Space,
    embedding_factories: Mapping[Type[gymnasium.Space], EmbeddingFactory] | None = None,
) -> Any:
    if embedding_factories is None:
        embedding_factories = get_default_embedding_factories()
    return gym_space_map(
        partial(_mk_embedding_fn, embedding_factories=embedding_factories), space
    )
