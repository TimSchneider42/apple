from __future__ import annotations

from abc import ABC, abstractmethod

import jax

from .attention_mask import AttentionMask


class AttentionFn(ABC):
    memory_alignment: int

    @abstractmethod
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
    ) -> jax.Array: ...
