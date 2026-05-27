from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from bounding_functions import (
    BoundingFunction,
    BoundRequirement,
    BOUNDING_FUNCTION_FACTORIES,
)

BoundedDenseGeneralType = Enum(
    "BoundedDenseGeneralType",
    {
        **{e.name: e.value for e in BoundingFunction},
        "L1_NORM": "l1_norm",
        "L2_NORM": "l2_norm",
        "INF_NORM": "inf_norm",
    },
    type=str,
)


class BoundedDenseGeneralFactory(ABC):
    def build(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *args,
        **kwargs,
    ) -> Callable[[jax.Array], jax.Array]:
        if np.any(lower > upper):
            raise ValueError(f"Lower bounds {lower} must be <= upper bounds {upper}")
        if not self.is_applicable(lower, upper).all():
            raise ValueError(
                f"Factory {self} is not applicable for bounds lower={lower}, upper={upper}"
            )
        return self._build(lower, upper, *args, **kwargs)

    @abstractmethod
    def _build(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *args,
        **kwargs,
    ) -> Callable[[jax.Array], jax.Array]:
        pass

    @abstractmethod
    def is_applicable(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        pass


@dataclass(frozen=True)
class CompositeBoundedDenseGeneralFactory(BoundedDenseGeneralFactory):
    factories: tuple[BoundedDenseGeneralFactory, ...]

    def _build(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *args,
        **kwargs,
    ) -> Callable[[jax.Array], jax.Array]:
        applicability_map = self.__get_applicability_map(lower, upper)
        indices = np.sum(
            np.cumprod(~applicability_map, axis=0).astype(np.bool_), axis=0
        )
        assert np.all(indices < len(self.factories))

        def fn(x: jax.Array) -> jax.Array:
            output = jnp.zeros(x.shape[:-1] + lower.shape, dtype=x.dtype)
            for i, f in enumerate(self.factories):
                mask = np.asarray(indices == i, dtype=bool)
                if np.any(mask):
                    output = output.at[..., mask].set(
                        f.build(lower[mask], upper[mask], *args, **kwargs)(x)
                    )

            return output

        return fn

    def is_applicable(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return np.all(np.any(self.__get_applicability_map(lower, upper), axis=0))

    def __get_applicability_map(self, lower: np.ndarray, upper: np.ndarray):
        return np.stack([f.is_applicable(lower, upper) for f in self.factories], axis=0)


@dataclass(frozen=True)
class SimpleBoundedDenseGeneralFactory(BoundedDenseGeneralFactory):
    bounding_fn: Callable[[np.ndarray, np.ndarray], Callable[[jax.Array], jax.Array]]
    lower_bound: BoundRequirement = BoundRequirement.NOT_SUPPORTED
    upper_bound: BoundRequirement = BoundRequirement.NOT_SUPPORTED

    def _build(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        *args,
        **kwargs,
    ) -> Callable[[jax.Array], jax.Array]:
        def fn(x: jax.Array) -> jax.Array:
            unbounded_value = nn.DenseGeneral(lower.shape, *args, **kwargs)(x)
            output = jnp.zeros_like(unbounded_value)
            non_inverted = self.__is_applicable(lower, upper)
            inverted = ~non_inverted
            output = output.at[..., non_inverted].set(
                self.bounding_fn(lower[non_inverted], upper[non_inverted])(
                    unbounded_value[..., non_inverted]
                )
            )
            output = output.at[..., inverted].set(
                -self.bounding_fn(-upper[inverted], -lower[inverted])(
                    -unbounded_value[..., inverted]
                )
            )
            return output

        return fn

    def __is_applicable(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        bounded_below = lower != -np.inf
        bounded_above = upper != np.inf

        supports_lower_bound = self.lower_bound != BoundRequirement.NOT_SUPPORTED
        supports_upper_bound = self.upper_bound != BoundRequirement.NOT_SUPPORTED

        needs_lower_bound = self.lower_bound == BoundRequirement.REQUIRED
        needs_upper_bound = self.upper_bound == BoundRequirement.REQUIRED

        below_ok = (bounded_below & supports_lower_bound) | (
            ~bounded_below & (not needs_lower_bound)
        )
        above_ok = (bounded_above & supports_upper_bound) | (
            ~bounded_above & (not needs_upper_bound)
        )

        return below_ok & above_ok

    def is_applicable(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        return self.__is_applicable(lower, upper) | self.__is_applicable(-upper, -lower)


BOUNDED_DENSE_GENERAL_FACTORIES = {
    BoundedDenseGeneralType(bounding_fn.value): SimpleBoundedDenseGeneralFactory(
        factory.mk_fn,
        lower_bound=factory.lower_bound,
        upper_bound=factory.upper_bound,
    )
    for bounding_fn, factory in BOUNDING_FUNCTION_FACTORIES.items()
}
