from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.flash_attention import (
    flash_attention,
    SegmentIds,
)

from .attention_fn import AttentionFn
from .attention_mask import AttentionMask


class JAXNativeFlashAttentionFn(AttentionFn):
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
                "Sowing weights is not supported in JAXNativeFlashAttentionFn."
            )
        if dropout_rate is not None and dropout_rate != 0.0:
            raise NotImplementedError(
                "Dropout is not supported in JAXNativeFlashAttentionFn."
            )

        if mask is None:
            mask = AttentionMask()

        batch_size, sequence_length, num_heads, head_dim = query.shape

        # Do not need episode mask because we provide segment ids and do not need the flat mask, because we fold it into
        # segment ids. Furthermore, the future horizon mask is only needed if is_causal is False.
        mask_tensor = mask.get_full_mask(
            sequence_length,
            include_same_episode_mask=False,
            include_flat_mask=False,
            include_future_horizon_mask=not mask.is_causal,
            return_as_float_mask=True,
        )

        if mask_tensor is not None:
            mask_tensor = jnp.broadcast_to(
                mask_tensor[..., None, :, :],
                (batch_size, num_heads, sequence_length, sequence_length),
            )

        if mask.episode_ids is None:
            segment_ids = jnp.zeros((batch_size, sequence_length), dtype=jnp.int32)
        else:
            segment_ids = mask.episode_ids

        if mask.flat_mask is not None:
            segment_ids = jnp.where(mask.flat_mask, segment_ids, -1)

        return flash_attention(
            query.swapaxes(1, 2),
            key.swapaxes(1, 2),
            value.swapaxes(1, 2),
            ab=mask_tensor,
            segment_ids=SegmentIds(segment_ids, segment_ids),
            causal=mask.is_causal,
            sm_scale=softmax_scaling_factor,
        ).swapaxes(1, 2)

    @property
    def memory_alignment(self) -> int:
        return 128
