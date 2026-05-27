from __future__ import annotations

from typing import Callable

import flax.struct
import jax.lax
import jax.numpy as jnp


@flax.struct.dataclass
class JaxRate:
    current_count: int
    interval: float
    include_zero: bool

    @classmethod
    def build(cls, interval: float, include_zero: bool = True):
        if interval < 0:
            raise ValueError("Interval must be non-negative.")
        return cls(current_count=0, interval=interval, include_zero=include_zero)

    def __call__(
        self, step: float | jax.Array, fn: Callable, skip_fn: Callable, *args, **kwargs
    ):
        return jax.lax.cond(
            self.due(step),
            lambda: (self.do(), fn(*args, **kwargs)),
            lambda: (self, skip_fn(*args, **kwargs)),
        )

    def do(self) -> "JaxRate":
        return self.replace(current_count=self.current_count + 1)

    def due(self, step: float | jax.Array) -> bool:
        return jax.lax.cond(
            self.interval == 0,
            lambda s: True,
            lambda s: s / self.interval + jnp.astype(self.include_zero, jnp.int32)
            > self.current_count,
            step,
        )

    def clear(self) -> "JaxRate":
        return self.replace(current_count=0)

    @classmethod
    def always(cls) -> "JaxRate":
        return cls.build(0)

    @classmethod
    def never(cls) -> "JaxRate":
        return cls.build(float("inf"), include_zero=False)
