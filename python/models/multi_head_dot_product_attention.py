from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, overload

import jax
import jax.numpy as jnp
from flax.linen import initializers, DenseGeneral
from flax.linen.linear import default_kernel_init
from flax.linen.module import Module, compact
from flax.typing import (
    PRNGKey,
    Dtype,
    Initializer,
    PrecisionLike,
    DotGeneralT,
)
from jax import lax

from .attention import AttentionFn, AttentionMask, FlaxDotProductAttentionFn
from .util import maybe_add_dims


class MultiHeadDotProductAttention(Module):
    """Multi-head dot-product attention.

    Adapted from `flax.linen.attention.MultiHeadDotProductAttention`.

    Attributes:
      num_heads: Number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: The dtype of the computation (default: infer from inputs and params)
      param_dtype: The dtype passed to parameter initializers (default: float32)
      qkv_features: Dimension of the key, query, and value.
      out_features: Dimension of the last projection
      kernel_init: Initializer for the kernel of the Dense layers.
      out_kernel_init: Optional Initializer for the kernel of the output Dense layer,
        if None, ``kernel_init`` will be used.
      bias_init: Initializer for the bias of the Dense layers.
      out_bias_init: Optional Initializer for the bias of the output Dense layer,
        if None, ``bias_init`` will be used.
      use_bias: Whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts query,
        key, value, and returns output of shape ``[bs, dim1, dim2, ..., dimN,,
        num_heads, value_channels]``
      decode: Whether to prepare and use an autoregressive cache.
    """

    num_heads: int
    dtype: Dtype | None = None
    param_dtype: Dtype = jnp.float32
    qkv_features: int | None = None
    out_features: int | None = None
    precision: PrecisionLike = None
    kernel_init: Initializer = default_kernel_init
    out_kernel_init: Initializer | None = None
    bias_init: Initializer = initializers.zeros_init()
    out_bias_init: Initializer | None = None
    use_bias: bool = True
    attention_fn: AttentionFn | None = None
    decode: bool = False
    # Deprecated, will be removed.
    qkv_dot_general: DotGeneralT | None = None
    out_dot_general: DotGeneralT | None = None
    qkv_dot_general_cls: Any = None
    out_dot_general_cls: Any = None
    q_dense_general: Callable[..., Callable[[jax.Array], jax.Array]] = DenseGeneral
    k_dense_general: Callable[..., Callable[[jax.Array], jax.Array]] = DenseGeneral
    v_dense_general: Callable[..., Callable[[jax.Array], jax.Array]] = DenseGeneral
    out_dense_general: Callable[..., Callable[[jax.Array], jax.Array]] = DenseGeneral
    softmax_scaling_factor: float | None = None

    @overload
    def __call__(
        self,
        inputs_q: jax.Array,
        inputs_k: jax.Array | None = None,
        inputs_v: jax.Array | None = None,
        *,
        mask: AttentionMask | None = None,
        evaluation_mode: bool = False,
        dropout_rate: float | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
    ): ...

    @overload
    def __call__(
        self,
        inputs_q: jax.Array,
        *,
        inputs_kv: jax.Array | None = None,
        mask: AttentionMask | None = None,
        evaluation_mode: bool = False,
        dropout_rate: float | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
    ): ...

    @compact
    def __call__(
        self,
        inputs_q: jax.Array,
        inputs_k: jax.Array | None = None,
        inputs_v: jax.Array | None = None,
        *,
        inputs_kv: jax.Array | None = None,
        mask: AttentionMask | None = None,
        evaluation_mode: bool = False,
        dropout_rate: float | None = None,
        dropout_rng: PRNGKey | None = None,
        sow_weights: bool = False,
    ):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        If both inputs_k and inputs_v are None, they will both copy the value of
        inputs_q (self attention).
        If only inputs_v is None, it will copy the value of inputs_k.

        Args:
          inputs_q: input queries of shape ``[batch_sizes..., length, features]``.
          inputs_k: key of shape ``[batch_sizes..., length, features]``. If None,
            inputs_k will copy the value of inputs_q.
          inputs_v: values of shape ``[batch_sizes..., length, features]``. If None,
            inputs_v will copy the value of inputs_k.
          inputs_kv: key/values of shape ``[batch_sizes..., length, features]``. If
            None, inputs_kv will copy the value of inputs_q. This arg will be
            deprecated soon. Use inputs_k and inputs_v instead.
          mask: attention mask of shape ``[batch_sizes..., num_heads, query_length,
            key/value_length]``. Attention weights are masked out if their
            corresponding mask value is ``False``.
          deterministic: if false, the attention weight is masked randomly using
            dropout, whereas if true, the attention weights are deterministic.
          dropout_rng: optional rng key to pass to the attention layer's dropout
            mask. Otherwise, self.make_rng('dropout') is used instead.
          sow_weights: if ``True``, the attention weights are sowed into the
            'intermediates' collection. Remember to mark 'intermediates' as
            mutable via ``mutable=['intermediates']`` in order to have that
            collection returned.

        Returns:
          output of shape ``[batch_sizes..., length, features]``.
        """
        if inputs_kv is not None:
            if inputs_k is not None or inputs_v is not None:
                raise ValueError(
                    "If either `inputs_k` or `inputs_v` is not None, "
                    "`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` "
                    "and `inputs_v` must be None. We recommend using `inputs_k` and "
                    "`inputs_v` args, since `inputs_kv` will be deprecated soon. See "
                    "https://github.com/google/flax/discussions/3389 for more "
                    "information."
                )
            inputs_k = inputs_v = inputs_kv
            warnings.warn(
                "The inputs_kv arg will be deprecated soon. "
                "Use inputs_k and inputs_v instead. See "
                "https://github.com/google/flax/discussions/3389 "
                "for more information.",
                DeprecationWarning,
            )
        else:
            if inputs_k is None:
                if inputs_v is not None:
                    raise ValueError(
                        "`inputs_k` cannot be None if `inputs_v` is not None. "
                        "To have both `inputs_k` and `inputs_v` be the same value, pass in the "
                        "value to `inputs_k` and leave `inputs_v` as None."
                    )
                inputs_k = inputs_q
            if inputs_v is None:
                inputs_v = inputs_k
            elif inputs_v.shape[-1] == inputs_v.shape[-2]:
                warnings.warn(
                    f"You are passing an array of shape {inputs_v.shape} "
                    "to the `inputs_v` arg, when you may have intended "
                    "to pass it to the `mask` arg. As of Flax version "
                    "0.7.4, the function signature of "
                    "MultiHeadDotProductAttention's `__call__` method "
                    "has changed to `__call__(inputs_q, inputs_k=None, "
                    "inputs_v=None, *, inputs_kv=None, mask=None, "
                    "deterministic=None)`. Use the kwarg `mask` instead. "
                    "See https://github.com/google/flax/discussions/3389 "
                    "and read the docstring for more information.",
                    DeprecationWarning,
                )

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            f"Memory dimension ({qkv_features}) must be divisible by number of"
            f" heads ({self.num_heads})."
        )
        head_dim = qkv_features // self.num_heads

        dense_kwargs = dict(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            use_bias=self.use_bias,
            precision=self.precision,
            dot_general=self.qkv_dot_general,
            dot_general_cls=self.qkv_dot_general_cls,
        )

        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]

        query = self.q_dense_general(**dense_kwargs, name="query")(inputs_q)
        key = self.k_dense_general(**dense_kwargs, name="key")(inputs_k)
        value = self.v_dense_general(**dense_kwargs, name="value")(inputs_v)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.decode:
            # detect if we're initializing by absence of existing cache data.
            is_initialized = self.has_variable("cache", "cached_key")
            cached_key = self.variable(
                "cache", "cached_key", jnp.zeros, key.shape, key.dtype
            )
            cached_value = self.variable(
                "cache", "cached_value", jnp.zeros, value.shape, value.dtype
            )
            cache_index = self.variable(
                "cache", "cache_index", lambda: jnp.array(0, dtype=jnp.int32)
            )
            if is_initialized:
                (
                    *batch_dims,
                    max_length,
                    num_heads,
                    depth_per_head,
                ) = cached_key.value.shape
                # shape check of cached keys against query input
                expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
                if expected_shape != query.shape:
                    raise ValueError(
                        "Autoregressive cache shape error, "
                        "expected query shape %s instead got %s."
                        % (expected_shape, query.shape)
                    )
                # update key, value caches with our new 1d spatial slices
                cur_index = cache_index.value
                zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
                indices: tuple[int | jax.Array, ...] = (zero,) * len(batch_dims) + (
                    cur_index,
                    zero,
                    zero,
                )
                key = lax.dynamic_update_slice(cached_key.value, key, indices)
                value = lax.dynamic_update_slice(cached_value.value, value, indices)
                cached_key.value = key
                cached_value.value = value
                cache_index.value = cache_index.value + 1
                # causal mask for cached decoder self-attention:
                # our single query position should only attend to those key
                # positions that have already been generated and cached,
                # not the remaining zero elements.
                new_mask = jnp.broadcast_to(
                    jnp.arange(max_length) <= cur_index,
                    tuple(batch_dims) + (1, max_length),
                )
                if mask is None:
                    mask = AttentionMask(base_mask=new_mask)
                else:
                    mask &= new_mask

        attention_fn = self.effective_attention_fn

        if inputs_q.shape[1] % self.memory_alignment != 0:
            raise ValueError(
                f"Sequence length of query must be a multiple of the memory alignment {attention_fn.memory_alignment} "
                f"(got {inputs_q.shape[1]})."
            )
        if inputs_k.shape[1] != inputs_v.shape[1]:
            raise ValueError(
                f"The sequence length of key and value must be the same (got {inputs_k.shape[1]} and "
                f"{inputs_v.shape[1]})."
            )
        if inputs_k.shape[1] % self.memory_alignment != 0:
            raise ValueError(
                f"Sequence length of key/value must be a multiple of the memory alignment "
                f"{attention_fn.memory_alignment} (got {inputs_k.shape[1]})."
            )

        x = attention_fn(
            query,
            key,
            value,
            mask=mask,
            softmax_scaling_factor=self.softmax_scaling_factor,
            evaluation_mode=evaluation_mode,
            sow_weights=sow_weights,
            dropout_rng=(
                self.make_rng("dropout") if dropout_rng is None else dropout_rng
            ),
            dropout_rate=dropout_rate,
        )
        # back to the original inputs dimensions
        out = self.out_dense_general(
            features=features,
            axis=(-2, -1),
            kernel_init=self.out_kernel_init or self.kernel_init,
            bias_init=self.out_bias_init or self.bias_init,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            dot_general=self.out_dot_general,
            dot_general_cls=self.out_dot_general_cls,
            name="out",  # type: ignore[call-arg]
        )(
            x,
        )
        return out

    @property
    def effective_attention_fn(self) -> AttentionFn:
        return (
            FlaxDotProductAttentionFn()
            if self.attention_fn is None
            else self.attention_fn
        )

    @property
    def memory_alignment(self):
        return self.effective_attention_fn.memory_alignment
