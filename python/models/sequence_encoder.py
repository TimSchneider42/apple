from __future__ import annotations

from abc import abstractmethod, ABC
from typing import Generic, TypeVar, Any

import flax.linen as nn
import jax

MemoryStateType = TypeVar("MemoryStateType")


class SequenceEncoder(nn.Module, Generic[MemoryStateType], ABC):
    @abstractmethod
    def __call__(
        self,
        sequence_starts: jax.Array,
        sequence: jax.Array,
        memory_state: MemoryStateType | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[jax.Array, MemoryStateType]:
        pass

    @abstractmethod
    def init_memory(self, batch_shape: tuple[int, ...]) -> MemoryStateType:
        pass

    @property
    def memory_alignment(self) -> int:
        return 1
