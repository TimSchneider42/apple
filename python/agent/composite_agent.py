from __future__ import annotations

from functools import partial
from typing import Any, Generic, TypeVar, Callable

import flax.linen as nn
import flax.struct
import jax

from models import MultiModalSequenceEncoder
from .base_agent import BaseAgent
from .cached import cached

CompositeAgentStructure = TypeVar("CompositeAgentStructure")


@flax.struct.dataclass
class ActObsTuple:
    action: Any
    observation: Any


class CompositeAgent(
    BaseAgent[CompositeAgentStructure, CompositeAgentStructure],
    Generic[CompositeAgentStructure],
):
    sequence_encoders: CompositeAgentStructure  # Inner type MultiModalSequenceEncoder
    heads: CompositeAgentStructure  # Inner type Callable[[jax.Array], jax.Array]
    observe_actions: bool = True
    enable_component: CompositeAgentStructure | None = None  # Inner type bool

    @nn.compact
    def __call__(
        self,
        observations: Any,
        prev_actions: jax.Array,
        sequence_starts: jax.Array,
        memory_state: CompositeAgentStructure | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        enable_component: CompositeAgentStructure | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[CompositeAgentStructure, CompositeAgentStructure]:
        if memory_state is None:
            memory_state = jax.tree.map(lambda s: None, self.sequence_encoders)

        # Override action with zeros for all batch dimensions where the sequence just starts
        actions_masked = jax.tree.map(
            lambda x: x
            * ~sequence_starts.reshape(
                sequence_starts.shape + (-1,) * (len(x.shape) - 2)
            ),
            prev_actions,
        )

        sequence = ActObsTuple(
            action=actions_masked if self.observe_actions else None,
            observation=observations,
        )

        # Evaluate the sequence encoders (we make sure to evaluate every encoder only once)
        encoder_output = self.cached_evaluation(
            lambda enc, mem: enc(
                sequence_starts,
                sequence,
                mem,
                step_no=step_no,
                evaluation_mode=evaluation_mode,
                norm_mask=norm_mask,
                max_output_step_count=max_output_step_count,
            ),
            memory_state,
            enable_component=enable_component,
        )

        hidden_state = jax.tree.map(
            lambda _, x: x[0], self.sequence_encoders, encoder_output
        )
        new_memory_state = jax.tree.map(
            lambda _, x: x[1], self.sequence_encoders, encoder_output
        )

        # Store the hidden states in the intermediates for debugging
        self.sow("intermediates", "hidden_state", hidden_state)

        output = jax.tree.map(
            lambda head, h: None if h is None else head(h), self.heads, hidden_state
        )

        return output, new_memory_state

    def component_enabled(
        self, enable_component: CompositeAgentStructure | None = None
    ):
        if self.is_initializing():
            return jax.tree.map(lambda e: True, self.sequence_encoders)
        else:
            if self.enable_component is None:
                enable_component_default = jax.tree.map(
                    lambda e: True, self.sequence_encoders
                )
            else:
                enable_component_default = self.enable_component
            if enable_component is None:
                return enable_component_default
            else:
                return jax.tree.map(
                    lambda e, ed: ed if e is None else e,
                    enable_component,
                    enable_component_default,
                )

    def init_memory(
        self,
        batch_shape: tuple[int, ...],
        enable_component: CompositeAgentStructure | None = None,
    ) -> CompositeAgentStructure:
        # Initialize memory only once per instance
        @partial(cached, key=lambda encoder: id(encoder))
        def init_mem(encoder: MultiModalSequenceEncoder):
            return encoder.init_memory(batch_shape)

        return jax.tree.map(
            lambda enc, en: init_mem(enc) if en else None,
            self.sequence_encoders,
            self.component_enabled(enable_component),
        )

    def cached_evaluation(
        self,
        fn: Callable,
        *args,
        enable_component: CompositeAgentStructure | None = None,
        **kwargs,
    ):
        @partial(cached, key=lambda encoder, *args, **kwargs: id(encoder))
        def eval_encoder(encoder: MultiModalSequenceEncoder, *args, **kwargs):
            return fn(encoder, *args, **kwargs)

        return jax.tree.map(
            lambda enc, en, *args, **kwargs: (
                eval_encoder(enc, *args, **kwargs) if en else None
            ),
            self.sequence_encoders,
            self.component_enabled(enable_component),
            *args,
            **kwargs,
        )

    def get_effective_sequence_encoders(self) -> CompositeAgentStructure:
        # Returns a CompositeAgentStructure of CompositeAgentStructures of bools, where the inner tree is True at the
        # position of the effective encoder
        return self.cached_evaluation(
            lambda encoder: jax.tree.map(
                lambda enc: enc is encoder, self.sequence_encoders
            )
        )
