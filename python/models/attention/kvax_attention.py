from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

jax.experimental.shard_map.Specs = Any

import kvax
import triton.language as tl
from kvax.ops.flash_attention_triton import flash_attention_triton_single_device
from kvax.ops.mask_creator import (
    compute_attention_mask_single_device,
    AttentionMask as KVAXAttentionMask,
)
from kvax.utils import PADDING_SEGMENT_ID, FlashAttentionParamsConfig
from kvax.utils.common import get_default_flash_attention_params

from .attention_fn import AttentionFn
from .attention_mask import AttentionMask, cached_class_method
import logging

logger = logging.getLogger(__name__)

# Workaround for kvax issue where these constants are not defined correctly
kvax.ops.flash_attention_triton.LOG2_CONST = tl.constexpr(1.4426950408889634)
kvax.ops.flash_attention_triton.NEG_INF = tl.constexpr(jnp.iinfo(jnp.int32).min)


@dataclass(frozen=True)
class KVAXMask:
    positions: jax.Array
    segment_ids: jax.Array
    mask_tensors: tuple[jax.Array, ...]


# Adapted from original KVAX code
def create_attention_mask_single_device(
    query_positions: jax.Array,
    query_segment_ids: jax.Array,
    kv_positions: jax.Array,
    kv_segment_ids: jax.Array,
    fwd_params: FlashAttentionParamsConfig | None = None,
    bwd_params: FlashAttentionParamsConfig | None = None,
    calc_bwd_mask: bool = False,
    skip_pad_tokens: bool = True,
) -> tuple[KVAXAttentionMask, ...]:
    # Define default parameters for flash attention if not provided.
    if fwd_params is None:
        fwd_params = get_default_flash_attention_params(backward=False)

    # Compute attention mask for forward kernel
    mask_fwd = compute_attention_mask_single_device(
        query_positions,
        query_segment_ids,
        kv_positions,
        kv_segment_ids,
        kv_block_size=fwd_params.kv_block_size,
        query_block_size=fwd_params.query_block_size,
        skip_pad_tokens=skip_pad_tokens,
    )
    outputs = [
        mask_fwd,
    ]

    # Compute attention mask for backward kernel if needed
    if calc_bwd_mask:
        if bwd_params is None:
            bwd_params = get_default_flash_attention_params(backward=True)

        query_block_size_dkdv = bwd_params.query_block_size
        kv_block_size_dkdv = bwd_params.kv_block_size
        query_block_size_dq = bwd_params.kv_block_size
        kv_block_size_dq = bwd_params.query_block_size

        # Compute attention mask for calculation dquery
        mask_dq = compute_attention_mask_single_device(
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            kv_block_size=kv_block_size_dq,
            query_block_size=query_block_size_dq,
            skip_pad_tokens=skip_pad_tokens,
        )

        # Compute attention mask for calculation dkey and dvalue
        mask_dkdv = compute_attention_mask_single_device(
            query_positions,
            query_segment_ids,
            kv_positions,
            kv_segment_ids,
            kv_block_size=kv_block_size_dkdv,
            query_block_size=query_block_size_dkdv,
            skip_pad_tokens=skip_pad_tokens,
            calculate_dkdv_mask=True,
        )

        outputs.append(mask_dq)
        outputs.append(mask_dkdv)

    return tuple(outputs)


# This is a bit of a hack, but it will essentially attach the cache to the AttentionMask instance.
@cached_class_method(maxsize=None)
def mk_kvax_mask(
    mask: AttentionMask, batch_size: int, sequence_length: int
) -> KVAXMask:
    positions = jnp.broadcast_to(
        jnp.arange(sequence_length)[None], (batch_size, sequence_length)
    )

    if mask.episode_ids is None:
        segment_ids = jnp.zeros_like(positions)
    else:
        segment_ids = mask.episode_ids

    if mask.flat_mask is not None:
        segment_ids = jnp.where(mask.flat_mask, segment_ids, PADDING_SEGMENT_ID)

    mask_tensors = create_attention_mask_single_device(
        positions, segment_ids, positions, segment_ids, calc_bwd_mask=True
    )
    return KVAXMask(
        positions=positions, segment_ids=segment_ids, mask_tensors=mask_tensors
    )


class KVAXDotProductAttentionFn(AttentionFn):
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
    ) -> jax.Array:
        if sow_weights is True:
            raise NotImplementedError(
                "Sowing weights is not supported in KVAXDotProductAttentionFn."
            )
        if dropout_rate is not None:
            if not isinstance(dropout_rate, float):
                logger.warning(
                    "Dynamic dropout rate ignored in KVAXDotProductAttentionFn."
                )
            elif dropout_rate != 0.0:
                raise NotImplementedError(
                    "Dropout is not supported in KVAXDotProductAttentionFn."
                )

        if query.shape[1] != key.shape[1]:
            raise NotImplementedError()

        if mask is None:
            mask = AttentionMask()

        kvax_mask = mk_kvax_mask(mask, query.shape[0], query.shape[1])

        return flash_attention_triton_single_device(
            query=query.transpose((0, 2, 1, 3)),
            key=key.transpose((0, 2, 1, 3)),
            value=value.transpose((0, 2, 1, 3)),
            query_positions=kvax_mask.positions,
            query_segment_ids=kvax_mask.segment_ids,
            kv_positions=kvax_mask.positions,
            kv_segment_ids=kvax_mask.segment_ids,
            mask_tensors=kvax_mask.mask_tensors,
            assume_sequential_positions=True,
            scale=softmax_scaling_factor,
        ).transpose((0, 2, 1, 3))

    @property
    def memory_alignment(self) -> int:
        return 64
