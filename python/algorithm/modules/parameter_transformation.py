from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Any

import flax.struct
import optax


class TransformInitFn(Protocol):
    def __call__(self, params: optax.Params) -> Any: ...


class TransformUpdateFn(Protocol):
    def __call__(
        self, updates: optax.Updates, state: Any, params: optax.Params
    ) -> tuple[optax.Params, Any]: ...


@dataclass(frozen=True)
class ParameterTransformation:
    init: TransformInitFn
    update: TransformUpdateFn


def simple_parameter_transformation(
    gradient_transformation: optax.GradientTransformation,
) -> ParameterTransformation:
    def init_fn(params: optax.Params) -> optax.OptState:
        return gradient_transformation.init(params)

    def update_fn(
        updates: optax.Updates,
        state: optax.OptState,
        params: optax.Params,
    ) -> tuple[optax.Params, optax.OptState]:
        updates, state = gradient_transformation.update(updates, state, params)
        return optax.apply_updates(params, updates), state

    return ParameterTransformation(init_fn, update_fn)


@flax.struct.dataclass
class ChainedGradientParameterTransformationState:
    gradient_transformation_state: optax.OptState
    parameter_transformation_state: optax.OptState


def chained_gradient_parameter_transformation(
    gradient_transformation: optax.GradientTransformation,
    parameter_transformation: ParameterTransformation,
) -> ParameterTransformation:
    def init_fn(params: optax.Params) -> ChainedGradientParameterTransformationState:
        return ChainedGradientParameterTransformationState(
            gradient_transformation.init(params), parameter_transformation.init(params)
        )

    def update_fn(
        updates: optax.Updates,
        state: ChainedGradientParameterTransformationState,
        params: optax.Params,
    ) -> tuple[optax.Params, ChainedGradientParameterTransformationState]:
        updates, gradient_transformation_state = gradient_transformation.update(
            updates, state.gradient_transformation_state, params
        )
        new_params, parameter_transformation_state = parameter_transformation.update(
            updates, state.parameter_transformation_state, params
        )
        return new_params, ChainedGradientParameterTransformationState(
            gradient_transformation_state, parameter_transformation_state
        )

    return ParameterTransformation(init_fn, update_fn)
