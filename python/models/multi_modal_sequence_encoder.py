from __future__ import annotations

from typing import Generic, Any

import flax.linen as nn
import jax

from .multi_modal_sequence_embedding import MultiModalSequenceEmbedding
from .sequence_encoder import SequenceEncoder, MemoryStateType


class MultiModalSequenceEncoder(nn.Module, Generic[MemoryStateType]):
    embedding: MultiModalSequenceEmbedding
    sequence_encoder: SequenceEncoder[MemoryStateType]

    @nn.compact
    def __call__(
        self,
        sequence_starts: jax.Array,
        input_sequences: dict[str, jax.Array],
        memory_state: MemoryStateType | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[jax.Array, MemoryStateType]:
        embeddings = self.embedding(
            input_sequences, evaluation_mode=evaluation_mode, norm_mask=norm_mask
        )

        # Store the embeddings in the intermediates for debugging
        self.sow("intermediates", "embeddings", embeddings)

        return self.sequence_encoder(
            sequence_starts,
            embeddings,
            memory_state=memory_state,
            step_no=step_no,
            evaluation_mode=evaluation_mode,
            norm_mask=norm_mask,
            max_output_step_count=max_output_step_count,
        )

    def init_memory(self, batch_shape: tuple[int, ...]) -> MemoryStateType:
        return self.sequence_encoder.init_memory(batch_shape)

    @property
    def memory_alignment(self) -> int:
        return self.sequence_encoder.memory_alignment
