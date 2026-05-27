from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from models import (
    Normalization,
    NORMALIZATION_FACTORIES,
    TransformerEncoder,
    DefaultMLPBlock,
    DefaultMultiHeadSelfAttentionBlock,
    TransformerBlock,
)
from models.attention import (
    FlaxDotProductAttentionFn,
    JAXNativeDotProductAttentionFn,
    JAXNativeFlashAttentionFn,
)

logger = logging.getLogger(__name__)


class AttentionFnImplementation(str, Enum):
    FLAX = "flax"
    XLA = "xla"
    CUDNN = "cudnn"
    JAX_FLASH = "jax_flash"
    KVAX = "kvax"
    AUTO = "auto"


@dataclass
class TransformerConfig:
    # Hidden dimensions
    hidden_dims: int

    # Number of heads
    num_heads: int

    # Which normalization to use
    normalization: Normalization

    # Number of layers
    layer_count: int

    # Which attention function implementation to use
    attention_fn: AttentionFnImplementation

    # Whether to use softmax scaling in the attention function
    apply_softmax_scaling: bool

    # Dropout rate to use in the attention blocks
    attention_dropout_rate: float

    def mk_transformer_mlp_block(self) -> TransformerBlock:
        return DefaultMLPBlock(
            dim_feedforward=self.hidden_dims,
            normalization=NORMALIZATION_FACTORIES[self.normalization],
        )

    def mk_transformer_encoder(
        self,
        min_sequence_length: int | None = None,
    ) -> TransformerEncoder:
        mlp_block = self.mk_transformer_mlp_block()

        attention_kwargs = dict(
            num_heads=self.num_heads,
        )

        def mk_attention_fn():
            attention_fn = self.attention_fn
            if attention_fn == AttentionFnImplementation.AUTO:
                # Choose the best available implementation.
                # If the memory horizon + context length is more than 65, KVAX may be chosen (below it has some sort of
                # bug that causes a crash).
                if (
                    min_sequence_length is not None
                    and min_sequence_length > 64
                    and self.attention_dropout_rate == 0.0
                ):
                    attention_fn = AttentionFnImplementation.KVAX
                else:
                    # use FLAX attention here as XLA does not seem to work that well
                    # TODO: investigate what is going on with XLA attention
                    attention_fn = AttentionFnImplementation.FLAX
            logger.info(f"Using {attention_fn.name} attention.")
            if attention_fn == AttentionFnImplementation.FLAX:
                return FlaxDotProductAttentionFn(
                    dropout_rate=self.attention_dropout_rate
                )
            elif attention_fn in {
                AttentionFnImplementation.XLA,
                AttentionFnImplementation.CUDNN,
            }:
                if self.attention_dropout_rate != 0.0:
                    raise NotImplementedError(
                        "Dropout is not supported in JAX native dot product attention."
                    )
                if attention_fn == AttentionFnImplementation.XLA:
                    implementation = "xla"
                else:
                    implementation = "cudnn"
                return JAXNativeDotProductAttentionFn(implementation=implementation)
            elif attention_fn == AttentionFnImplementation.KVAX:
                if self.attention_dropout_rate != 0.0:
                    raise NotImplementedError(
                        "Dropout is not supported in KVAX attention."
                    )
                from models.attention.kvax_attention import KVAXDotProductAttentionFn

                return KVAXDotProductAttentionFn()
            elif attention_fn == AttentionFnImplementation.JAX_FLASH:
                if self.attention_dropout_rate != 0.0:
                    raise NotImplementedError(
                        "Dropout is not supported in JAX native flash attention."
                    )
                return JAXNativeFlashAttentionFn()
            else:
                raise NotImplementedError(
                    f"Unknown attention function type '{attention_fn}'"
                )

        attention_block = DefaultMultiHeadSelfAttentionBlock(
            **attention_kwargs,
            normalization=NORMALIZATION_FACTORIES[self.normalization],
            apply_softmax_scaling=self.apply_softmax_scaling,
            attention_fn=mk_attention_fn(),
        )

        return TransformerEncoder(
            layers=sum(
                [
                    [attention_block.clone(), mlp_block.clone()]
                    for _ in range(self.layer_count)
                ],
                [],
            ),
        )
