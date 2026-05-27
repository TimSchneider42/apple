from __future__ import annotations

import math

import distrax
import jax
import jax.numpy as jnp


class ConstrainedNormal(distrax.Distribution):
    BOUNDS_OPTIONAL = False

    def __init__(
        self,
        loc: jax.Array,
        scale: jax.Array,
        low: jax.Array = -jnp.inf,
        high: jax.Array = jnp.inf,
    ):
        self.__low = low
        self.__high = high
        self.__loc = loc
        self.__scale = scale

    @classmethod
    def build(
        cls,
        loc: jax.Array,
        scale: jax.Array,
        low: jax.Array = -jnp.inf,
        high: jax.Array = jnp.inf,
    ) -> "ConstrainedNormal":
        if not cls.BOUNDS_OPTIONAL and (jnp.isinf(low).any() or jnp.isinf(high).any()):
            raise NotImplementedError(
                f"{cls.__name__} requires lower and upper bounds to be finite."
            )
        return cls(loc, scale, low, high)

    @property
    def low(self):
        return self.__low

    @property
    def high(self):
        return self.__high

    @property
    def loc(self):
        return self.__loc

    @property
    def scale(self):
        return self.__scale

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.__low.shape


class SACConstrainedNormal(ConstrainedNormal):
    """
    Idea taken from cleanrl: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py
    """

    def __init__(
        self, loc: jax.Array, scale: jax.Array, low: jax.Array, high: jax.Array
    ):
        super().__init__(loc, scale, low, high)
        self.__inner = distrax.Normal(loc, scale)

    def _sample_n(self, key: jax.Array, n: int) -> jax.Array:
        hidden = self.__inner._sample_n(key, n)
        return jnp.tanh(hidden) * self.__scale + self.__bias

    def log_prob(self, value: jax.Array) -> jax.Array:
        value_norm = (value - self.__bias) / self.__scale
        value_norm_clipped = jnp.clip(value_norm, -1 + 1e-6, 1 - 1e-6)
        hidden = jnp.arctanh(value_norm_clipped)
        return self.__log_prob(self.__inner.log_prob(hidden), value_norm_clipped)

    def _sample_n_and_log_prob(
        self, key: jax.Array, n: int
    ) -> tuple[jax.Array, jax.Array]:
        hidden, log_prob = self.__inner._sample_n_and_log_prob(key, n)
        value_norm = jnp.tanh(hidden)
        return value_norm * self.__scale + self.__bias, self.__log_prob(
            log_prob, value_norm
        )

    def __log_prob(self, inner_log_prob: jax.Array, value_norm: jax.Array) -> jax.Array:
        return inner_log_prob - jnp.log(
            self.__scale * (1 - jnp.clip(value_norm, -1, 1) ** 2) + 1e-6
        )

    @property
    def __scale(self):
        return (self.high - self.low) / 2

    @property
    def __bias(self):
        return (self.high + self.low) / 2


class TruncatedNormal(ConstrainedNormal):
    BOUNDS_OPTIONAL = True

    def __init__(
        self, loc: jax.Array, scale: jax.Array, low: jax.Array, high: jax.Array
    ):
        super().__init__(loc, scale, low, high)
        self.__inner = distrax.Normal(loc, scale)
        self.__log_offset = -jnp.log(
            self.__inner.cdf(self.high) - self.__inner.cdf(self.low)
        ) - jnp.log(self.scale)

    def _sample_n(self, key: jax.Array, n: int) -> jax.Array:
        lower_rel = (self.low - self.loc) / self.scale
        upper_rel = (self.high - self.loc) / self.scale
        inner_sample = jax.random.truncated_normal(
            key, lower_rel[None], upper_rel[None], (n,) + self.loc.shape
        )
        return inner_sample * self.scale + self.loc

    def log_prob(self, value: jax.Array) -> jax.Array:
        normal_log_prob = self.__inner.log_prob(value)
        return normal_log_prob + self.__log_offset

    def entropy(self) -> jax.Array:
        # See https://en.wikipedia.org/wiki/Truncated_normal_distribution
        normal_dist = distrax.Normal(self.loc, self.scale)
        z = normal_dist.cdf(self.high) - normal_dist.cdf(self.low)
        alpha = (self.low - self.loc) / self.scale
        beta = (self.high - self.loc) / self.scale
        return jnp.log(math.sqrt(2 * math.pi * math.e) * self.scale * z) + (
            alpha * normal_dist.prob(self.low) - beta * normal_dist.prob(self.high)
        ) / (2 * z)
