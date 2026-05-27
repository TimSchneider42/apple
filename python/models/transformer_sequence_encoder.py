from __future__ import annotations

import math
from typing import Callable, Any

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp

from ring_buffer import RingBuffer
from .normalization import NoNorm, BaseNormalization
from .sequence_encoder import SequenceEncoder
from .attention import AttentionMask
from .transformer import TransformerEncoder
from .util import maybe_add_dims


@flax.struct.dataclass
class TransformerSequenceStep:
    value: jax.Array
    step_no: jax.Array
    norm_mask: jax.Array


TransformerMemoryState = RingBuffer[TransformerSequenceStep]


class PositionalEncoding(nn.Module):
    input_dims: int
    dropout_rate: float = 0.1
    max_len: int = 5000
    target_std: jax.Array | float | None = None

    @nn.compact
    def __call__(
        self, embeddings: jax.Array, step_no: jax.Array, evaluation_mode: bool = False
    ) -> jax.Array:
        dropout = nn.Dropout(rate=self.dropout_rate)

        position = jnp.arange(self.max_len)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.input_dims, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / self.input_dims)
        )
        pe = jnp.zeros((self.max_len, self.input_dims))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(
            jnp.cos((position * div_term)[:, : self.input_dims // 2])
        )
        if self.target_std is None:
            scale = 1.0
        else:
            pe_std = 1 / math.sqrt(2)
            scale = self.target_std / pe_std
        return dropout(embeddings + pe[step_no] * scale, deterministic=evaluation_mode)


class TransformerSequenceEncoder(SequenceEncoder[TransformerMemoryState]):
    transformer_encoder: TransformerEncoder
    output_dims: int | None = None
    dropout_rate: float = 0.1
    embedding_dims: int = 2048
    scale_inputs: bool = True
    encode_inputs: bool = True
    memory_horizon: int = 16
    can_see_future: bool = False
    encoding_normalization: Callable[[], BaseNormalization] = NoNorm
    attention_ignore_memory_horizon: bool = False
    use_positional_encoding: bool = True
    max_episode_length: int = 5000

    @nn.compact
    def __call__(
        self,
        sequence_starts: jax.Array,
        sequence: jax.Array,
        memory_state: TransformerMemoryState | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[jax.Array, TransformerMemoryState]:
        batch_shape = sequence.shape[:-2]
        step_count = sequence.shape[-2]
        if max_output_step_count is not None:
            step_count = min(step_count, max_output_step_count)

        # This is a hack to be JIT compatible
        self.variable(
            "params",
            "input_dims_dummy",
            lambda s: s[(0,) * (len(s.shape) - 1)],
            sequence,
        )

        memory_alignment = self.memory_alignment
        memory_size_aligned = (
            math.ceil(self.memory_horizon / memory_alignment) * memory_alignment
        )
        step_count_aligned = self.next_aligned_value(step_count)

        # Fast pass, avoiding an expensive jnp.roll operation on the memory state
        fast_pass = (
            memory_state is None
            and max(step_count_aligned, memory_size_aligned) == step_count
        )
        if memory_state is None:
            memory_state = self.__init_memory(
                batch_shape, max(step_count_aligned, memory_size_aligned)
            )
        elif memory_state.capacity < step_count_aligned:
            memory_state = memory_state.resize(self.next_aligned_value(step_count))

        # We do this to avoid using jax.lax.cond, which is inefficient sometimes
        cond_int = (memory_state.length > 0).astype(jnp.int32)
        last_step_no = memory_state[memory_state.length - 1].step_no * cond_int + (
            1 - cond_int
        ) * jnp.full(batch_shape, -1)

        if step_no is None:
            new_sequence_pos = (
                last_step_no[:, None] + 1 + jnp.arange(sequence_starts.shape[-1])[None]
            )
            new_sequence_start_pos = jax.lax.cummax(
                sequence_starts * new_sequence_pos, axis=len(batch_shape)
            )
            step_no = new_sequence_pos - new_sequence_start_pos

        if norm_mask is None:
            norm_mask = jnp.ones(sequence.shape[:2], dtype=jnp.bool_)

        new_memory_state = memory_state.add(
            TransformerSequenceStep(sequence, step_no, norm_mask)
        )

        if fast_pass:
            seq = TransformerSequenceStep(sequence, step_no, norm_mask)
        else:
            seq = new_memory_state.data_aligned
        # CAREFUL: Not all elements in the sequence are valid, as the data_aligned contains all elements of the ring
        #          buffer, even unused ones. We remove invalid elements at the end of the function.

        # We need to mask out the data that is not valid (beyond the occupied space of the buffer)
        data_valid_mask = (jnp.arange(seq.step_no.shape[-1]) < new_memory_state.length)[
            None
        ]

        full_norm_mask = seq.norm_mask
        full_norm_mask &= data_valid_mask

        # Fold data_valid_mask into episode ids
        episode_id = jnp.where(
            data_valid_mask, jnp.cumsum(seq.step_no == 0, axis=-1), -1
        )

        # -1 is correct here because the current step is counted in the memory horizon
        if self.attention_ignore_memory_horizon:
            attention_horizon = None
        else:
            attention_horizon = self.memory_horizon - 1
        attention_mask = AttentionMask(
            episode_ids=episode_id,
            past_horizon=attention_horizon,
            future_horizon=attention_horizon if self.can_see_future else 0,
        )

        embedding_dims = self.embedding_dims
        if not self.encode_inputs:
            embedding_dims = max(embedding_dims, sequence.shape[-1])
        num_heads = self.transformer_encoder.num_heads
        embedding_dims = int(math.ceil(embedding_dims / num_heads)) * num_heads
        if self.encode_inputs:
            seq_enc = nn.Dense(
                embedding_dims,
                name="input_encoder",
            )(seq.value)
            seq_enc = self.encoding_normalization()(seq_enc)
        else:
            pad_width = ((0, 0),) * len(sequence_starts.shape) + (
                (0, embedding_dims - sequence.shape[-1]),
            )
            seq_enc = self.encoding_normalization()(
                jnp.pad(sequence, pad_width),
                evaluation_mode=evaluation_mode,
                mask=maybe_add_dims(full_norm_mask, 1),
            )

        if self.scale_inputs:
            # Might not be needed: https://github.com/espnet/espnet/issues/2797
            seq_enc *= math.sqrt(embedding_dims)

        if self.use_positional_encoding:
            pos_encoder = PositionalEncoding(
                embedding_dims,
                self.dropout_rate,
                max_len=self.max_episode_length,
            )
            seq_enc = pos_encoder(
                seq_enc,
                seq.step_no,
                evaluation_mode=evaluation_mode,
            )

        # Store the input sequence with positional encoding in the intermediates for debugging
        self.sow("intermediates", "seq_enc", seq_enc)

        transformer_output = self.transformer_encoder(
            seq_enc,
            attention_mask,
            src_pos=seq.step_no,
            evaluation_mode=evaluation_mode,
            norm_mask=full_norm_mask,
        )

        if self.output_dims is None:
            output = transformer_output
        else:
            output = nn.Dense(self.output_dims)(transformer_output)

        if fast_pass:
            output_slice = output
        else:
            output_slice = jax.lax.dynamic_slice_in_dim(
                output,
                new_memory_state.length - step_count,
                step_count,
                axis=len(output.shape) - 2,
            )
        if new_memory_state.capacity != memory_size_aligned:
            new_memory_state = new_memory_state.resize(memory_size_aligned)
        return output_slice, new_memory_state

    @property
    def input_dims(self) -> int:
        return self.variables["params"]["input_dims_dummy"].shape[0]

    def __init_memory(
        self, batch_shape: tuple[int, ...], memory_horizon: int
    ) -> TransformerMemoryState:
        spec = TransformerSequenceStep(
            jax.ShapeDtypeStruct((self.input_dims,), dtype=jnp.float32),
            jax.ShapeDtypeStruct((), dtype=jnp.int32),
            jax.ShapeDtypeStruct((), dtype=jnp.bool_),
        )
        return RingBuffer.build(spec, batch_shape=batch_shape, capacity=memory_horizon)

    def init_memory(self, batch_shape: tuple[int, ...]) -> TransformerMemoryState:
        return self.__init_memory(
            batch_shape, self.next_aligned_value(self.memory_horizon)
        )

    def next_aligned_value(self, value: int) -> int:
        memory_alignment = self.memory_alignment
        return math.ceil(value / memory_alignment) * memory_alignment

    @property
    def memory_alignment(self):
        return self.transformer_encoder.memory_alignment
