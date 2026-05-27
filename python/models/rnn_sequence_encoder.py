from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import RNNCellBase

from .sequence_encoder import SequenceEncoder


class RNNSequenceEncoder(SequenceEncoder[jax.Array]):
    rnn_cell: RNNCellBase = nn.GRUCell(features=64)
    embedding_net_hidden_units: tuple[int, ...] = (64, 64)

    @nn.compact
    def __call__(
        self,
        sequence_starts: jax.Array,
        sequence: jax.Array,
        memory_state: jax.Array | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        batch_shape = sequence_starts.shape[:-1]

        if memory_state is None:
            # For some reason the initialize_carry function requires an extra dimension at the end
            memory_state = self.init_memory(batch_shape)

        input_embedding = nn.Sequential(
            [e for h in self.embedding_net_hidden_units for e in [nn.Dense(h), nn.relu]]
        )(sequence)

        def single_step(mod, carry, t):
            mem_state, rng = carry
            rng, init_rng = jax.random.split(rng)
            new_carry = mod.initialize_carry(init_rng, batch_shape + (1,))
            masked_mem_state = jax.tree.map(
                lambda m, nc: jnp.where(sequence_starts[..., t, None], nc, m),
                mem_state,
                new_carry,
            )
            new_memory, output = mod(masked_mem_state, input_embedding[..., t, :])
            return (new_memory, rng), output

        # Compute the memory sequence
        # The mask resets memory for all batch dimensions where the sequence just starts
        (output_memory_state, _), output_sequence = nn.scan(
            single_step,
            variable_broadcast="params",
            split_rngs={"params": False},
            out_axes=1,
        )(
            self.rnn_cell,
            (memory_state, self.make_rng("carry")),
            jnp.arange(input_embedding.shape[-2]),
        )

        if max_output_step_count is not None:
            output_sequence = output_sequence[:, :max_output_step_count]

        return output_sequence, output_memory_state

    def init_memory(self, batch_shape: tuple[int, ...]) -> jax.Array:
        return self.rnn_cell.initialize_carry(
            self.make_rng("carry"), batch_shape + (1,)
        )
