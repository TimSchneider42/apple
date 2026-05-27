from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Type

import distrax
import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.linear import default_kernel_init

from algorithm.wrappers import Discrete32
from bounding_functions import (
    BoundingFunction,
)
from distribution_tree import DistributionTree
from gym_space_map import gym_space_map
from models import BOUNDED_DENSE_GENERAL_FACTORIES, CompositeBoundedDenseGeneralFactory
from tree_wrapper import TreeWrapper
from .constrained_normal import ConstrainedNormal

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def warn_once(msg: str):
    logger.warning(msg)


class ActorHead(nn.Module):
    action_space: gymnasium.Space
    std_allowed_bounding_fns: tuple[BoundingFunction, ...] = tuple(BoundingFunction)
    std_use_log_scale: bool = True
    log_std_min: float | None = -5
    log_std_max: float | None = 2
    mean_allowed_bounding_fns: tuple[BoundingFunction, ...] = tuple(BoundingFunction)
    act_bound_mean: bool = False
    kernel_init: nn.initializers.Initializer = default_kernel_init
    constrained_normal_type: Type[ConstrainedNormal] | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        if isinstance(self.action_space, Discrete32):
            return distrax.Categorical(
                nn.Dense(self.action_space.n, kernel_init=self.kernel_init)(x)
            )
        elif isinstance(self.action_space, gymnasium.spaces.Box):
            if self.act_bound_mean:
                mean_bounded_dense_general_factory = (
                    CompositeBoundedDenseGeneralFactory(
                        tuple(
                            BOUNDED_DENSE_GENERAL_FACTORIES[f]
                            for f in self.mean_allowed_bounding_fns
                        )
                    )
                )

                loc = mean_bounded_dense_general_factory.build(
                    np.broadcast_to(self.action_space.low, self.action_space.shape),
                    np.broadcast_to(self.action_space.high, self.action_space.shape),
                    kernel_init=self.kernel_init,
                )(x)
            else:
                loc = nn.DenseGeneral(
                    self.action_space.shape, kernel_init=self.kernel_init
                )(x)

            if (
                self.log_std_max is not None
                and self.log_std_min is not None
                and self.log_std_max <= self.log_std_min
            ):
                scale = np.exp(self.log_std_min)
            else:
                std_bounded_dense_general_factory = CompositeBoundedDenseGeneralFactory(
                    tuple(
                        BOUNDED_DENSE_GENERAL_FACTORIES[f]
                        for f in self.std_allowed_bounding_fns
                    )
                )

                log_std_min_val = (
                    -np.inf if self.log_std_min is None else self.log_std_min
                )
                log_std_max_val = (
                    np.inf if self.log_std_max is None else self.log_std_max
                )

                if self.std_use_log_scale:
                    scale = jnp.exp(
                        std_bounded_dense_general_factory.build(
                            np.full(self.action_space.shape, log_std_min_val),
                            np.full(self.action_space.shape, log_std_max_val),
                            kernel_init=self.kernel_init,
                        )(x)
                    )
                else:
                    scale = std_bounded_dense_general_factory.build(
                        np.full(self.action_space.shape, np.exp(self.log_std_min)),
                        np.full(self.action_space.shape, np.exp(self.log_std_max)),
                        kernel_init=self.kernel_init,
                    )(x)

            any_finite = np.any(np.isfinite(self.action_space.low)) or np.any(
                np.isfinite(self.action_space.high)
            )
            all_finite = np.all(np.isfinite(self.action_space.low)) and np.all(
                np.isfinite(self.action_space.high)
            )
            if self.constrained_normal_type is not None:
                if self.constrained_normal_type.BOUNDS_OPTIONAL or all_finite:
                    return self.constrained_normal_type(
                        loc,
                        scale,
                        jnp.array(self.action_space.low),
                        jnp.array(self.action_space.high),
                    )
                if any_finite:
                    warn_once(
                        "Using unconstrained normal distribution for action space with partial bounds "
                        f"because {self.constrained_normal_type.__name__} does not support partial bounds."
                    )
            else:
                if any_finite:
                    warn_once(
                        "Using unconstrained normal distribution for action space with bounds because "
                        "no constrained normal distribution type was specified."
                    )
            return distrax.Normal(loc, scale)
        else:
            raise NotImplementedError(f"Unsupported action space {self.action_space}")


class ActorHeadTree(TreeWrapper["ActorHeadTree", ActorHead], nn.Module):
    def __call__(self, x: jax.Array) -> distrax.Distribution:
        return DistributionTree.wrap(jax.tree.map(lambda head: head(x), self.orig_tree))

    @classmethod
    def is_leaf(cls, x: Any) -> bool:
        return isinstance(x, ActorHead)

    @classmethod
    def from_action_space(
        cls,
        action_space: gymnasium.Space,
        std_allowed_bounding_fns: tuple[BoundingFunction, ...] = tuple(
            BoundingFunction
        ),
        std_use_log_scale: bool = True,
        log_std_min: float | None = -5,
        log_std_max: float | None = 2,
        mean_allowed_bounding_fns: tuple[BoundingFunction, ...] = tuple(
            BoundingFunction
        ),
        act_bound_mean: bool = False,
        kernel_init: nn.initializers.Initializer = default_kernel_init,
        constrained_normal_type: Type[ConstrainedNormal] | None = None,
    ) -> "ActorHeadTree | ActorHead":
        return ActorHeadTree.wrap(
            gym_space_map(
                lambda a: ActorHead(
                    a,
                    std_allowed_bounding_fns=std_allowed_bounding_fns,
                    std_use_log_scale=std_use_log_scale,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    mean_allowed_bounding_fns=mean_allowed_bounding_fns,
                    act_bound_mean=act_bound_mean,
                    kernel_init=kernel_init,
                    constrained_normal_type=constrained_normal_type,
                ),
                action_space,
            )
        )
