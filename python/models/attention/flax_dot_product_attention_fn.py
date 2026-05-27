from __future__ import annotations

import functools
import inspect
import math
from collections.abc import Callable

import jax
import flax.linen as nn
from flax.linen.dtypes import promote_dtype
from flax.typing import Dtype, PrecisionLike
import jax.numpy as jnp

from .attention_fn import AttentionFn
from .attention_mask import AttentionMask


# Taken and adapted from the Flax code
def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jax.Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: nn.Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., jax.Array] | None = None,
    qk_attn_weights_einsum: Callable[..., jax.Array] | None = None,
    attn_weights_value_einsum: Callable[..., jax.Array] | None = None,
):
    """Computes dot-product attention given query, key, and value.

    This is the core function for applying attention based on
    https://arxiv.org/abs/1706.03762. It calculates the attention weights given
    query and key and combines the values using the attention weights.

    .. note::
      ``query``, ``key``, ``value`` needn't have any batch dimensions.

    Args:
      query: queries for calculating attention with shape of ``[batch...,
        q_length, num_heads, qk_depth_per_head]``.
      key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
      value: values to be used in attention with shape of ``[batch..., kv_length,
        num_heads, v_depth_per_head]``.
      bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``False``.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs)
      precision: numerical precision of the computation see ``jax.lax.Precision`
        for details.
      module: the Module that will sow the attention weights into the
        'intermediates' collection. Remember to mark 'intermediates' as mutable
        via ``mutable=['intermediates']`` in order to have that collection
        returned. If ``module`` is None, the attention weights will not be sowed.
      force_fp32_for_softmax: bool, whether to force the softmax to be computed in
        fp32. This is useful for mixed-precision training where higher precision
        is desired for numerical stability.
      einsum_dot_general: the dot_general to use in `jnp.einsum`.
      qk_attn_weights_einsum: the einsum for computing the attention weights. When
        unspecified, the default `jnp.einsum` will be used. This argument is
        mutually exclusive with `precision` and `einsum_dot_general`.
      attn_weights_value_einsum: the einsum for computing the product of the
        attention weights and the values. When unspecified, the default
        `jnp.einsum` will be used. This argument is mutually exclusive with
        `precision` and `einsum_dot_general`.

    Returns:
      Output of shape ``[batch..., q_length, num_heads, v_depth_per_head]``.

    Raises:
      ValueError: if both `precision`/`einsum_dot_general` and
      `qk_attn_weights_einsum`/`attn_weights_value_einsum` are
        specified.
    """
    if (qk_attn_weights_einsum and not attn_weights_value_einsum) or (
        not qk_attn_weights_einsum and attn_weights_value_einsum
    ):
        raise ValueError(
            "qk_attn_weights_einsum and attn_weights_value_einsum must be specified"
            " together."
        )
    if (precision or einsum_dot_general) and (
        qk_attn_weights_einsum or attn_weights_value_einsum
    ):
        raise ValueError(
            "precision/einsum_dot_general and"
            " qk_attn_weights_einsum/attn_weights_value_einsum are mutually"
            " exclusive. Please specify only one of them."
        )

    query, key, value = promote_dtype(query, key, value, dtype=dtype)
    dtype = query.dtype
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert (
        query.shape[:-3] == key.shape[:-3] == value.shape[:-3]
    ), "q, k, v batch dims must match."
    assert (
        query.shape[-2] == key.shape[-2] == value.shape[-2]
    ), "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    # compute attention weights
    attn_weights = dot_product_attention_weights(
        query,
        key,
        bias,
        mask,
        broadcast_dropout,
        dropout_rng,
        dropout_rate,
        deterministic,
        dtype,
        precision,
        module,
        force_fp32_for_softmax,
        einsum_dot_general=einsum_dot_general,
        einsum=qk_attn_weights_einsum,
    )
    if not attn_weights_value_einsum:
        attn_weights_value_einsum = functools.partial(
            jnp.einsum,
            precision=precision,
            _dot_general=(
                einsum_dot_general if einsum_dot_general else jax.lax.dot_general
            ),
        )
    # return weighted sum over values for each query position
    return attn_weights_value_einsum(
        "...hqk,...khd->...qhd",
        attn_weights,
        value,
    )


def dot_product_attention_weights(
    query: jax.Array,
    key: jax.Array,
    bias: jax.Array | None = None,
    mask: jax.Array | None = None,
    broadcast_dropout: bool = True,
    dropout_rng: jax.Array | None = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Dtype | None = None,
    precision: PrecisionLike = None,
    module: nn.Module | None = None,
    force_fp32_for_softmax: bool = False,
    einsum_dot_general: Callable[..., jax.Array] | None = None,
    einsum: Callable[..., jax.Array] | None = None,
):
    """Computes dot-product attention weights given query and key.

    Used by :func:`dot_product_attention`, which is what you'll most likely use.
    But if you want access to the attention weights for introspection, then
    you can directly call this function and call einsum yourself.

    Args:
      query: queries for calculating attention with shape of ``[batch...,
        q_length, num_heads, qk_depth_per_head]``.
      key: keys for calculating attention with shape of ``[batch..., kv_length,
        num_heads, qk_depth_per_head]``.
      bias: bias for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks, padding masks, proximity bias, etc.
      mask: mask for the attention weights. This should be broadcastable to the
        shape ``[batch..., num_heads, q_length, kv_length]``. This can be used for
        incorporating causal masks. Attention weights are masked out if their
        corresponding mask value is ``False``.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      dtype: the dtype of the computation (default: infer from inputs and params)
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      module: the Module that will sow the attention weights into the
        'intermediates' collection. Remember to mark 'intermediates' as mutable
        via ``mutable=['intermediates']`` in order to have that collection
        returned. If ``module`` is None, the attention weights will not be sowed.
      force_fp32_for_softmax: bool, whether to force the softmax to be computed in
        fp32. This is useful for mixed-precision training where higher precision
        is desired for numerical stability.
      einsum_dot_general: the dot_general to use in einsum.
      einsum: If unspecified, default `jnp.einsum` will be used. This argument is
        mutually exclusive with `precision` and `einsum_dot_general`.

    Raises:
      ValueError: if both `precision`/`einsum_dot_general` and `einsum` are
        specified.

    Returns:
      Output of shape ``[batch..., num_heads, q_length, kv_length]``.
    """
    if (precision or einsum_dot_general) and einsum:
        raise ValueError(
            "precision/einsum_dot_general and einsum are mutually exclusive. Please"
            " specify only one of them."
        )
    if not einsum:
        einsum = functools.partial(
            jnp.einsum,
            precision=precision,
            _dot_general=(
                einsum_dot_general if einsum_dot_general else jax.lax.dot_general
            ),
        )

    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype

    assert query.ndim == key.ndim, "q, k must have same rank."
    assert query.shape[:-3] == key.shape[:-3], "q, k batch dims must match."
    assert query.shape[-2] == key.shape[-2], "q, k num_heads must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = einsum("...qhd,...khd->...hqk", query, key)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
        attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    # normalize the attention weights
    if force_fp32_for_softmax and dtype != jnp.float32:
        attn_weights = jax.nn.softmax(attn_weights.astype(jnp.float32))
    else:
        attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    if module:
        module.sow("intermediates", "attention_weights", attn_weights)

    # apply attention dropout
    if not deterministic:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
            keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
        multiplier = keep.astype(dtype) / (jnp.asarray(keep_prob, dtype=dtype) + 1e-6)
        attn_weights = attn_weights * multiplier

    return attn_weights


class FlaxDotProductAttentionFn(nn.Module, AttentionFn):
    dtype: Dtype | None = None
    broadcast_dropout: bool = True
    dropout_rate: float = 0.0
    precision: PrecisionLike = None
    force_fp32_for_softmax: bool = False
    qk_attn_weights_einsum_cls: Callable[..., Callable[..., jax.Array]] | None = None
    attn_weights_value_einsum_cls: Callable[..., Callable[..., jax.Array]] | None = None

    def __call__(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
        mask: AttentionMask | None = None,
        softmax_scaling_factor: float | None = None,
        evaluation_mode: bool = False,
        sow_weights: bool = False,
        dropout_rate: float | None = None,
        dropout_rng: jax.Array | None = None,
    ):
        if softmax_scaling_factor is not None:
            undo_softmax_scaling = math.sqrt(query.shape[-1])
            scale = softmax_scaling_factor * undo_softmax_scaling
            query *= scale

        dropout_rate = (
            jnp.maximum(self.dropout_rate, dropout_rate)
            if dropout_rate is not None
            else self.dropout_rate
        )

        if (
            not isinstance(dropout_rate, float) or dropout_rate > 0.0
        ):  # Require `deterministic` only if using dropout.
            dropout_deterministic = evaluation_mode
            if not dropout_deterministic and dropout_rng is None:
                dropout_rng = self.make_rng("dropout")
            attention_fn = dot_product_attention
        else:
            dropout_deterministic = True
            attention_fn = nn.attention.dot_product_attention

        # `qk_attn_weights_einsum` and `attn_weights_value_einsum` are optional
        # arguments that can be used to override the default `jnp.einsum`. They
        # exist for quantized einsum support in AQT.
        qk_attn_weights_einsum = (
            self.qk_attn_weights_einsum_cls()
            if self.qk_attn_weights_einsum_cls
            else None
        )
        attn_weights_value_einsum = (
            self.attn_weights_value_einsum_cls()
            if self.attn_weights_value_einsum_cls
            else None
        )
        # apply attention
        attn_args = (query, key, value)
        # This kwargs list match the default nn.dot_product_attention.
        # For custom `attention_fn`s, invalid kwargs will be filtered.
        attn_kwargs = dict(
            mask=(
                None
                if mask is None
                else mask.get_full_mask(query.shape[1])[..., None, :, :]
            ),
            dropout_rng=dropout_rng,
            dropout_rate=dropout_rate,
            broadcast_dropout=self.broadcast_dropout,
            deterministic=dropout_deterministic,
            dtype=self.dtype,
            precision=self.precision,
            force_fp32_for_softmax=self.force_fp32_for_softmax,
            qk_attn_weights_einsum=qk_attn_weights_einsum,
            attn_weights_value_einsum=attn_weights_value_einsum,
        )
        attn_kwargs = {
            k: v
            for k, v in attn_kwargs.items()
            if k in inspect.signature(attention_fn).parameters
        }
        if sow_weights:
            return attention_fn(*attn_args, **attn_kwargs, module=self)
        else:
            return attention_fn(*attn_args, **attn_kwargs)

    @property
    def memory_alignment(self) -> int:
        return 1
