from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np


class BoundRequirement(str, Enum):
    NOT_SUPPORTED = "not_supported"
    OPTIONAL = "optional"
    REQUIRED = "required"


class BoundingFunction(str, Enum):
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SIN = "sin"
    SOFTPLUS = "softplus"
    ABS = "abs"
    NONE = "none"


@dataclass(frozen=True)
class BoundingFunctionFactory:
    mk_fn: Callable[
        [np.ndarray | float, np.ndarray | float], Callable[[jax.Array], jax.Array]
    ]
    mk_inverse_fn: Callable[
        [np.ndarray | float, np.ndarray | float], Callable[[jax.Array], jax.Array]
    ]
    lower_bound: BoundRequirement
    upper_bound: BoundRequirement

    @property
    def flip(self):
        return BoundingFunctionFactory(
            lambda l, u: lambda x: -self.mk_fn(-u, -l)(-x),
            lambda l, u: lambda x: -self.mk_inverse_fn(-u, -l)(-x),
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
        )


BOUNDING_FUNCTION_FACTORIES = {
    # We are scaling x here to ensure that the derivative of x at 0 remains 1.
    BoundingFunction.TANH: BoundingFunctionFactory(
        lambda l, u: lambda x: l + (jax.nn.tanh(x * 2 / (u - l)) + 1) / 2 * (u - l),
        lambda l, u: lambda y: jnp.arctanh(2 * (y - l) / (u - l) - 1) / 2 * (u - l),
        lower_bound=BoundRequirement.REQUIRED,
        upper_bound=BoundRequirement.REQUIRED,
    ),
    BoundingFunction.SIN: BoundingFunctionFactory(
        lambda l, u: lambda x: l + (jnp.sin(x * 2 / (u - l)) + 1) / 2 * (u - l),
        lambda l, u: lambda y: jnp.arcsin(2 * (y - l) / (u - l) - 1) / 2 * (u - l),
        lower_bound=BoundRequirement.REQUIRED,
        upper_bound=BoundRequirement.REQUIRED,
    ),
    BoundingFunction.SIGMOID: BoundingFunctionFactory(
        lambda l, u: lambda x: l + jax.nn.sigmoid(4 * x / (u - l)) * (u - l),
        lambda l, u: lambda y: jnp.log((y - l) / (u - y)) / 4 * (u - l),
        lower_bound=BoundRequirement.REQUIRED,
        upper_bound=BoundRequirement.REQUIRED,
    ),
    BoundingFunction.SOFTPLUS: BoundingFunctionFactory(
        lambda l, u: lambda x: l + jax.nn.softplus(x),
        lambda l, u: lambda y: jnp.log(jnp.expm1(y - l)),
        lower_bound=BoundRequirement.REQUIRED,
        upper_bound=BoundRequirement.NOT_SUPPORTED,
    ),
    BoundingFunction.ABS: BoundingFunctionFactory(
        lambda l, u: lambda x: l + jnp.abs(x),
        lambda l, u: lambda y: y - l,
        lower_bound=BoundRequirement.REQUIRED,
        upper_bound=BoundRequirement.NOT_SUPPORTED,
    ),
    BoundingFunction.NONE: BoundingFunctionFactory(
        lambda l, u: lambda x: x,
        lambda l, u: lambda x: x,
        lower_bound=BoundRequirement.NOT_SUPPORTED,
        upper_bound=BoundRequirement.NOT_SUPPORTED,
    ),
}


def get_bounding_fn_factory(
    lower: float, upper: float, allowed_fns: Iterable[BoundingFunctionFactory]
) -> BoundingFunctionFactory:
    """Get a bounding function factory for the specified range.

    Args:
        lower: The lower bound of the output range.
        upper: The upper bound of the output range.
        allowed_fns: An iterable of allowed bounding functions to choose from.

    Returns:
        A callable bounding function factory.

    Raises:
        ValueError: If no suitable bounding function factory is found for the given bounds.
    """
    if lower >= upper:
        raise ValueError(f"Lower bound {lower} must be less than upper bound {upper}.")

    allowed_factories = [*allowed_fns, *[fn.flip for fn in allowed_fns]]

    for factory in allowed_factories:
        has_upper = not np.isinf(upper)
        has_lower = not np.isinf(lower)

        needs_upper = factory.upper_bound == BoundRequirement.REQUIRED
        needs_lower = factory.lower_bound == BoundRequirement.REQUIRED

        supports_upper = factory.upper_bound != BoundRequirement.NOT_SUPPORTED
        supports_lower = factory.lower_bound != BoundRequirement.NOT_SUPPORTED

        upper_ok = has_upper and supports_upper or (not has_upper and not needs_upper)
        lower_ok = has_lower and supports_lower or (not has_lower and not needs_lower)

        if upper_ok and lower_ok:
            return factory

    raise ValueError(
        f"No suitable bounding function found for bounds ({lower}, {upper}) "
        f"from allowed functions: {allowed_fns}"
    )
