from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

import flax.linen as nn
import flax.typing
import jax

from .batch_norm import BatchRenorm as BatchRenormBase, BatchNorm as BatchNormBase
from .scaling_initializer import ScalingInitializer


class BaseNormalization(nn.Module, ABC):
    @abstractmethod
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        pass


class BatchNorm(BaseNormalization):
    axis: flax.typing.Axes = -1
    use_bias: bool = True
    use_scale: bool = True
    constant_scale: ScalingInitializer | float = 1.0
    momentum: float = 0.99
    stop_stat_gradients: bool = False
    time_axis: int | None = None

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        return BatchNormBase(
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            axis=self.axis,
            momentum=self.momentum,
            stop_stat_gradients=self.stop_stat_gradients,
            time_axis=self.time_axis,
        )(
            x, use_running_average=evaluation_mode, mask=mask
        ) * self.__get_constant_scale(
            x.shape
        )

    def __get_constant_scale(self, shape: tuple[int, ...]) -> float:
        if isinstance(self.constant_scale, float):
            return self.constant_scale
        return self.constant_scale(shape)


class BatchRenorm(BaseNormalization):
    axis: flax.typing.Axes = -1
    use_bias: bool = True
    use_scale: bool = True
    constant_scale: ScalingInitializer | float = 1.0
    warmup_steps: int = 100_000

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        return BatchRenormBase(
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            axis=self.axis,
            warmup_steps=self.warmup_steps,
        )(
            x, use_running_average=evaluation_mode, mask=mask
        ) * self.__get_constant_scale(
            x.shape
        )

    def __get_constant_scale(self, shape: tuple[int, ...]) -> float:
        if isinstance(self.constant_scale, float):
            return self.constant_scale
        return self.constant_scale(shape)


class LayerNorm(BaseNormalization):
    axis: flax.typing.Axes = -1
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        # Important: Ignore the mask, as this norm does not track statistics and including it will lead to NaNs because
        # if an entire vector is masked out the mean variance of that vector cannot be computed.
        return nn.LayerNorm(
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            reduction_axes=self.axis,
        )(x)


class NoNorm(BaseNormalization):
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        return x


class Normalization(str, Enum):
    BATCH_NORM = "batch_norm"
    BATCH_RENORM = "batch_renorm"
    LAYER_NORM = "layer_norm"
    NONE = "none"


NORMALIZATION_FACTORIES = {
    Normalization.BATCH_NORM: BatchNorm,
    Normalization.BATCH_RENORM: BatchRenorm,
    Normalization.LAYER_NORM: LayerNorm,
    Normalization.NONE: lambda axis=None: NoNorm(),
}
