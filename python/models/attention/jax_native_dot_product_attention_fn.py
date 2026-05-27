from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from .attention_fn import AttentionFn
from .attention_mask import AttentionMask


class JAXNativeDotProductAttentionFn(AttentionFn):
    def __init__(self, implementation: Literal["xla", "cudnn"] | None = None):
        if implementation is None:
            implementation = "xla"
        self.__implementation = implementation
        if self.__implementation == "xla":
            self.__memory_alignment = 1
        else:
            self.__memory_alignment = 64

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
        if sow_weights is True:
            raise NotImplementedError(
                "Sowing weights is not supported in JAXNativeDotProductAttentionFn."
            )
        if dropout_rate is not None and dropout_rate != 0.0:
            raise NotImplementedError(
                "Dropout is not supported in JAXNativeDotProductAttentionFn."
            )

        if mask is None:
            mask = AttentionMask()

        local_window_size = None
        if mask.past_horizon is not None or mask.future_horizon is not None:
            window_size_past = (
                mask.past_horizon if mask.past_horizon is not None else query.shape[-3]
            )
            window_size_future = (
                mask.future_horizon
                if mask.future_horizon is not None
                else query.shape[-3]
            )
            local_window_size = (window_size_past, window_size_future)

        if self.__implementation == "xla":
            query_conv = query
            key_conv = key
            value_conv = value
        else:
            # cuDNN requires float16 inputs.
            query_conv = query.astype(jnp.float16)
            key_conv = key.astype(jnp.float16)
            value_conv = value.astype(jnp.float16)

        # Do not need horizon masks here because we set is_causal and define the window
        mask_tensor = mask.get_full_mask(
            query.shape[1],
            include_past_horizon_mask=False,
            include_future_horizon_mask=False,
        )

        if mask_tensor is not None:
            mask_tensor = mask_tensor[..., None, :, :]

        return jax.nn.dot_product_attention(
            query_conv,
            key_conv,
            value_conv,
            mask=mask_tensor,
            is_causal=mask.is_causal,
            local_window_size=local_window_size,
            implementation=self.__implementation,
            scale=softmax_scaling_factor,
        ).astype(value.dtype)

    @property
    def memory_alignment(self) -> int:
        return self.__memory_alignment
