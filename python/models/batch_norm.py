from __future__ import annotations

from typing import Any, Callable, Sequence, Union

import flax.typing
import jax
import jax.numpy as jnp
from flax.linen.module import Module, compact, merge_param
from flax.linen.normalization import _canonicalize_axes, _compute_stats, _normalize
from jax.nn import initializers

PRNGKey = Any
Array = Any
Shape = tuple[int, ...]
Dtype = Any
Axes = Union[int, Sequence[int]]


# Adapted from flax.linen.normalization.BatchNorm
class BatchNorm(Module):
    """BatchNorm Module.

    Usage Note:
    If we define a model with BatchNorm, for example::

      >>> import flax.linen as nn
      >>> import jax, jax.numpy as jnp
      >>> BN = nn.BatchNorm(momentum=0.9, epsilon=1e-5, dtype=jnp.float32)

    The initialized variables dict will contain, in addition to a 'params'
    collection, a separate 'batch_stats' collection that will contain all the
    running statistics for all the BatchNorm layers in a model::

      >>> x = jax.random.normal(jax.random.key(0), (5, 6))
      >>> variables = BN.init(jax.random.key(1), x, use_running_average=False)
      >>> jax.tree_util.tree_map(jnp.shape, variables)
      {'batch_stats': {'mean': (6,), 'var': (6,)}, 'params': {'bias': (6,), 'scale': (6,)}}

    We then update the batch_stats during training by specifying that the
    ``batch_stats`` collection is mutable in the ``apply`` method for our
    module.::

      >>> y, new_batch_stats = BN.apply(variables, x, mutable=['batch_stats'], use_running_average=False)

    During eval we would define BN with ``use_running_average=True`` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn't mark it
    mutable::

      >>> y = BN.apply(variables, x, mutable=['batch_stats'], use_running_average=True)

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See ``jax.pmap`` for a description of axis names (default: None).
        Note, this is only used for pmap and shard map. For SPMD jit, you do not
        need to manually synchronize. Just make sure that the axes are correctly
        annotated and XLA:SPMD will insert the necessary collectives.
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, ``[[0, 1], [2, 3]]`` would independently batch-normalize over the
        examples on the first two and last two devices. See ``jax.lax.psum`` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: bool | None = None
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: flax.typing.Initializer = initializers.zeros
    scale_init: flax.typing.Initializer = initializers.ones
    axis_name: str | None = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True
    force_float32_reductions: bool = True
    stop_stat_gradients: bool = False
    time_axis: int | None = None

    @compact
    def __call__(
        self,
        x,
        use_running_average: bool | None = None,
        *,
        mask: jax.Array | None = None,
    ):
        """Normalizes the input using batch statistics.

        .. note::
          During initialization (when ``self.is_initializing()`` is ``True``) the running
          average of the batch statistics will not be updated. Therefore, the inputs
          fed during initialization don't need to match that of the actual input
          distribution and the reduction axis (set with ``axis_name``) does not have
          to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.
          mask: Binary array of shape broadcastable to ``inputs`` tensor, indicating
            the positions for which the mean and variance should be computed.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(
                s,
                jnp.float32 if self.force_float32_reductions else self.param_dtype,
            ),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats",
            "var",
            lambda s: jnp.ones(
                s,
                jnp.float32 if self.force_float32_reductions else self.param_dtype,
            ),
            feature_shape,
        )

        if use_running_average:
            mean = (
                ra_mean.value
                if self.force_float32_reductions
                else jnp.asarray(ra_mean.value, self.param_dtype)
            )
            var = (
                ra_var.value
                if self.force_float32_reductions
                else jnp.asarray(ra_var.value, self.param_dtype)
            )
        else:
            x_stats = x
            if self.time_axis is not None:
                # Select only one sample from each time sequence to prevent information from leaking between time steps.
                index_array = jax.random.randint(
                    self.make_rng("batch_norm"),
                    shape=(
                        *x.shape[: self.time_axis],
                        1,
                        *x.shape[self.time_axis + 1 :],
                    ),
                    minval=0,
                    maxval=x.shape[self.time_axis],
                )
                x_stats = jnp.take_along_axis(x_stats, index_array, axis=self.time_axis)
                mask = (
                    None
                    if mask is None
                    else jnp.take_along_axis(mask, index_array, axis=self.time_axis)
                )
            mean, var = _compute_stats(
                x_stats,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
                force_float32_reductions=self.force_float32_reductions,
            )

            if not self.is_initializing():
                ra_mean.value = (
                    self.momentum * ra_mean.value + (1 - self.momentum) * mean
                )
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        if self.stop_stat_gradients:
            mean = jax.lax.stop_gradient(mean)
            var = jax.lax.stop_gradient(var)

        return _normalize(
            self,
            x,
            mean,
            var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
            self.force_float32_reductions,
        )


# Adapted from https://github.com/araffin/sbx/blob/master/sbx/common/jax_layers.py
class BatchRenorm(Module):
    """BatchRenorm Module (https://arxiv.org/abs/1702.03275).
    Adapted from flax.linen.normalization.BatchNorm

    BatchRenorm is an improved version of vanilla BatchNorm. Contrary to BatchNorm,
    BatchRenorm uses the running statistics for normalizing the batches after a warmup phase.
    This makes it less prone to suffer from "outlier" batches that can happen
    during very long training runs and, therefore, is more robust during long training runs.

    During the warmup phase, it behaves exactly like a BatchNorm layer.

    Usage Note:
    If we define a model with BatchRenorm, for example::

      BRN = BatchRenorm(use_running_average=False, momentum=0.99, epsilon=0.001, dtype=jnp.float32)

    The initialized variables dict will contain in addition to a 'params'
    collection a separate 'batch_stats' collection that will contain all the
    running statistics for all the BatchRenorm layers in a model::

      vars_initialized = BRN.init(key, x)  # {'params': ..., 'batch_stats': ...}

    We then update the batch_stats during training by specifying that the
    `batch_stats` collection is mutable in the `apply` method for our module.::

      vars_in = {'params': params, 'batch_stats': old_batch_stats}
      y, mutated_vars = BRN.apply(vars_in, x, mutable=['batch_stats'])
      new_batch_stats = mutated_vars['batch_stats']

    During eval we would define BRN with `use_running_average=True` and use the
    batch_stats collection from training to set the statistics.  In this case
    we are not mutating the batch statistics collection, and needn't mark it
    mutable::

      vars_in = {'params': params, 'batch_stats': training_batch_stats}
      y = BRN.apply(vars_in, x)

    Attributes:
      use_running_average: if True, the statistics stored in batch_stats will be
        used. Else the running statistics will be first updated and then used to normalize.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of the batch
        statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  if True, bias (beta) is added.
      use_scale: if True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over the
        examples on the first two and last two devices. See `jax.lax.psum` for
        more details.
      use_fast_variance: If true, use a faster, but less numerically stable,
        calculation for the variance.
    """

    use_running_average: bool | None = None
    axis: flax.typing.Axes = -1
    momentum: float = 0.99
    epsilon: float = 0.001
    warmup_steps: int = 100_000
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: str | None = None
    axis_index_groups: Any = None
    use_fast_variance: bool = True
    stop_stat_gradients: bool = False

    @compact
    def __call__(
        self,
        x,
        *,
        use_running_average: bool | None = None,
        mask: jax.Array | None = None,
    ):
        """Normalizes the input using batch statistics.

        NOTE:
        During initialization (when `self.is_initializing()` is `True`) the running
        average of the batch statistics will not be updated. Therefore, the inputs
        fed during initialization don't need to match that of the actual input
        distribution and the reduction axis (set with `axis_name`) does not have
        to exist.

        Args:
          x: the input to be normalized.
          use_running_average: if true, the statistics stored in batch_stats will be
            used instead of computing the batch statistics on the input.

        Returns:
          Normalized inputs (the same shape as inputs).
        """

        use_running_average = merge_param(
            "use_running_average", self.use_running_average, use_running_average
        )
        feature_axes = _canonicalize_axes(x.ndim, self.axis)
        reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
        feature_shape = [x.shape[ax] for ax in feature_axes]

        ra_mean = self.variable(
            "batch_stats",
            "mean",
            lambda s: jnp.zeros(s, jnp.float32),
            feature_shape,
        )
        ra_var = self.variable(
            "batch_stats", "var", lambda s: jnp.ones(s, jnp.float32), feature_shape
        )

        r_max = self.variable(
            "batch_stats",
            "r_max",
            lambda s: s,
            3.0,
        )
        d_max = self.variable(
            "batch_stats",
            "d_max",
            lambda s: s,
            5.0,
        )
        steps = self.variable(
            "batch_stats",
            "steps",
            lambda s: s,
            0.0,
        )

        if use_running_average:
            custom_mean = ra_mean.value
            custom_var = ra_var.value
        else:
            batch_mean, batch_var = _compute_stats(
                x,
                reduction_axes,
                dtype=self.dtype,
                axis_name=self.axis_name if not self.is_initializing() else None,
                axis_index_groups=self.axis_index_groups,
                use_fast_variance=self.use_fast_variance,
                mask=mask,
            )
            if self.is_initializing():
                custom_mean = batch_mean
                custom_var = batch_var
            else:
                std = jnp.sqrt(batch_var + self.epsilon)
                ra_std = jnp.sqrt(ra_var.value + self.epsilon)
                # scale
                r = jax.lax.stop_gradient(std / ra_std)
                r = jnp.clip(r, 1 / r_max.value, r_max.value)
                # bias
                d = jax.lax.stop_gradient((batch_mean - ra_mean.value) / ra_std)
                d = jnp.clip(d, -d_max.value, d_max.value)

                # BatchNorm normalization, using minibatch stats and running average stats
                # Because we use _normalize, this is equivalent to
                # ((x - x_mean) / sigma) * r + d = ((x - x_mean) * r + d * sigma) / sigma
                # where sigma = sqrt(var)
                affine_mean = batch_mean - d * jnp.sqrt(batch_var) / r
                affine_var = batch_var / (r**2)

                # Note: in the original paper, after some warmup phase (batch norm phase of 5k steps)
                # the constraints are linearly relaxed to r_max/d_max over 40k steps
                # Here we only have a warmup phase
                is_warmed_up = jnp.greater_equal(steps.value, self.warmup_steps).astype(
                    jnp.float32
                )
                custom_mean = (
                    is_warmed_up * affine_mean + (1.0 - is_warmed_up) * batch_mean
                )
                custom_var = (
                    is_warmed_up * affine_var + (1.0 - is_warmed_up) * batch_var
                )

                ra_mean.value = (
                    self.momentum * ra_mean.value + (1.0 - self.momentum) * batch_mean
                )
                ra_var.value = (
                    self.momentum * ra_var.value + (1.0 - self.momentum) * batch_var
                )
                steps.value += 1

        if self.stop_stat_gradients:
            custom_mean = jax.lax.stop_gradient(custom_mean)
            custom_var = jax.lax.stop_gradient(custom_var)

        return _normalize(
            self,
            x,
            custom_mean,
            custom_var,
            reduction_axes,
            feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
        )
