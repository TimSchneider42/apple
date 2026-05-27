from __future__ import annotations

from typing import Union, Sequence, Callable, Any

import flax.linen as nn
import jax
from flax.linen.linear import default_kernel_init

from .normalization import BaseNormalization, NoNorm
from .sequence_encoder import SequenceEncoder
from .util import maybe_add_dims


class SimpleSequenceEncoder(SequenceEncoder[None]):
    act_fn: Callable[[jax.Array], jax.Array] = nn.tanh
    hidden_layers: int = 2
    hidden_dims: int = 64
    output_dims: int | None = None
    kernel_init: Union[
        nn.initializers.Initializer, Sequence[nn.initializers.Initializer]
    ] = default_kernel_init
    normalization: Callable[[], BaseNormalization] = NoNorm

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
        total_layer_count = self.hidden_layers + (
            1 if self.output_dims is not None else 0
        )
        kernel_init = self.kernel_init
        if not isinstance(kernel_init, Sequence):
            kernel_init = [kernel_init] * total_layer_count
        assert len(kernel_init) == total_layer_count
        output = sequence
        output = self.normalization()(
            output, evaluation_mode=evaluation_mode, mask=maybe_add_dims(norm_mask, 1)
        )
        for i in range(total_layer_count):
            final_layer = i == total_layer_count - 1 and self.output_dims is not None

            if final_layer:
                normalization = NoNorm()
            else:
                normalization = self.normalization()

            output = nn.Dense(
                self.hidden_dims,
                kernel_init=kernel_init[i],
            )(
                output,
                evaluation_mode=evaluation_mode,
                norm_mask=maybe_add_dims(norm_mask, 1),
            )
            output = normalization(output)

            if not final_layer:
                output = self.act_fn(output)
        if max_output_step_count is not None:
            output = output[:, :max_output_step_count]
        return output, None

    def init_memory(self, batch_shape: tuple[int, ...]) -> None:
        return None
