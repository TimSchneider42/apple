from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax

from models import SequenceEncoder


class IdSequenceEncoder(SequenceEncoder[None]):
    @nn.compact
    def __call__(
        self,
        sequence_starts: jax.Array,
        sequence: jax.Array,
        memory_state: None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[jax.Array, None]:
        if max_output_step_count is not None:
            sequence = sequence[:, :max_output_step_count]
        return sequence, None

    def init_memory(self, batch_shape: tuple[int, ...]) -> None:
        return None
