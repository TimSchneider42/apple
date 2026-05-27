from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any

import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jax_util import compute_param_metrics, metrics_set_valid_flag, path_to_str
from .parameter_transformation import (
    ParameterTransformation,
    simple_parameter_transformation,
    chained_gradient_parameter_transformation,
)
from .schedule_config import ScheduleConfig

logger = logging.getLogger(__name__)


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    SF_ADAMW = "sf_adamw"
    RMSPROP = "rmsprop"


@flax.struct.dataclass
class GradientMetricsState:
    metrics: dict[str, Any]


def compute_update_metrics(name: str) -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> GradientMetricsState:
        return GradientMetricsState(
            metrics={
                name: metrics_set_valid_flag(
                    compute_param_metrics(
                        f"mean_abs_value",
                        jax.tree.map(lambda _: np.nan, params),
                    ),
                    override_empty=True,
                    zeros_like_fn=np.zeros_like,
                )
            },
        )

    def update_fn(
        updates: optax.Updates,
        state: GradientMetricsState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, GradientMetricsState]:
        state = GradientMetricsState(
            metrics={
                name: metrics_set_valid_flag(
                    compute_param_metrics(
                        f"mean_abs_value",
                        jax.tree.map(lambda u: jnp.abs(u).mean(), updates),
                    ),
                    zeros_like_fn=np.zeros_like,
                )
            },
        )
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)


def filter_gradient_probes() -> optax.GradientTransformation:
    def init_fn(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: optax.Params | None = None,
    ) -> tuple[optax.Updates, optax.EmptyState]:
        return (
            jax.tree.map_with_path(
                lambda p, x: (
                    jnp.zeros_like(x)
                    if len(p) > 0 and p[-1].key == "__gradient_probe__"
                    else x
                ),
                updates,
            ),
            state,
        )

    return optax.GradientTransformation(init_fn, update_fn)


@dataclass
class OptimizerConfig:
    # The learning rate of the optimizer
    learning_rate: ScheduleConfig

    # Which optimizer to use. Choices: adam, adamw, sgd, sf_adamw, rmsprop
    type: OptimizerType

    # Whether to use a binary step (-1, 1) for the optimizer. Essentially chaining a signum function to the optimizer
    binary_step: bool

    # The epsilon value for the ADAM optimizer
    adam_eps: float

    # The b1 value for the ADAM optimizer
    adam_b1: float

    # The momentum for the SGD optimizer
    sgd_momentum: float | None

    # The maximum norm for the gradient clipping
    max_grad_norm: float

    def make_optimizer(
        self, total_step_count: float, track_metrics: bool = False
    ) -> ParameterTransformation:
        gradient_transformations = []
        if track_metrics:
            gradient_transformations.append(compute_update_metrics("raw_gradient"))
        if not np.isinf(self.max_grad_norm):
            gradient_transformations.append(
                optax.clip_by_global_norm(self.max_grad_norm)
            )
            if track_metrics:
                gradient_transformations.append(
                    compute_update_metrics("clipped_gradient")
                )

        if self.type == OptimizerType.ADAM:
            optimizer_transformation = partial(
                optax.adam, eps=self.adam_eps, b1=self.adam_b1
            )
        elif self.type == OptimizerType.ADAMW:
            optimizer_transformation = partial(optax.adamw, eps=self.adam_eps)
        elif self.type == OptimizerType.SGD:
            optimizer_transformation = partial(optax.sgd, momentum=self.sgd_momentum)
        elif self.type == OptimizerType.SF_ADAMW:
            optimizer_transformation = partial(
                optax.contrib.schedule_free_adamw, eps=self.adam_eps
            )
        elif self.type == OptimizerType.RMSPROP:
            optimizer_transformation = optax.rmsprop
        else:
            raise NotImplementedError(f"Unknown optimizer {self.type}")
        gradient_transformations.append(
            optax.inject_hyperparams(optimizer_transformation)(
                self.learning_rate.make_schedule(total_step_count)
            )
        )
        if track_metrics:
            gradient_transformations.append(
                compute_update_metrics("post_optimizer_updates")
            )
        if self.binary_step:
            gradient_transformations.append(
                optax.GradientTransformation(
                    lambda p: optax.EmptyState(),
                    lambda u, s, p: (jax.tree.map(jnp.sign, u), s),
                ),
            )
        gradient_transformations.append(filter_gradient_probes())
        gradient_transformation = optax.chain(*gradient_transformations)
        return simple_parameter_transformation(gradient_transformation)


def _get_metrics(node: Any) -> Any:
    if isinstance(node, dict):
        return node.get("metrics", None)
    else:
        return getattr(node, "metrics", None)


def _has_metrics(node: Any) -> bool:
    return _get_metrics(node) is not None


def extract_optimizer_metrics(optimizer_state: optax.OptState) -> dict[str, Any]:
    optimizer_metrics = [
        _get_metrics(s)
        for s in jax.tree.flatten(optimizer_state, is_leaf=_has_metrics)[0]
        if _has_metrics(s)
    ]

    metrics = {}
    for m in optimizer_metrics:
        assert not any(k in metrics for k in m.keys()), "Duplicate metric keys found"
        metrics.update(m)
    return metrics
