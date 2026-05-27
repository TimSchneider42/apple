from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TypeVar, Generic

import distrax
import flax.linen as nn
import jax

from pytree import PyTree


class BaseAgentOutput(ABC):
    @property
    @abstractmethod
    def action_distr(self) -> distrax.Distribution:
        pass

    @property
    @abstractmethod
    def prediction(self) -> PyTree[jax.Array]:
        pass


AgentOutputType = TypeVar("AgentOutputType", bound=BaseAgentOutput)
AgentMemoryType = TypeVar("AgentMemoryType")


class BaseAgent(nn.Module, ABC, Generic[AgentOutputType, AgentMemoryType]):
    @abstractmethod
    def __call__(
        self,
        observations: jax.Array,
        actions: jax.Array,
        sequence_starts: jax.Array,
        memory_state: AgentMemoryType | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[AgentOutputType, AgentMemoryType]:
        """
        Executes a sequence of steps with the given observations, actions and sequence-start bits. Note that this
        function is not affected by the agent's state and also does not change it.

        In the following, we denote
        B: batch dimensions, can be multidimensional
        S: sequence length (1D)
        O: observation dimensions (1D)
        A: action dimensions (1D)
        :param observations:            A B x S x O tensor containing the observations for each time step
        :param actions:                 A B x S x A tensor containing the actions for each time step. Note that
                                        actions[..., t] is the action that lead to observations[..., t].
        :param sequence_starts:         A B x S tensor that is true wherever a sequence starts.
        :param memory_state:            The memory state of the agent. If None, the agent assumes the episode just
                                        started and initializes its memory state accordingly.
        :param step_no:                 A B x S tensor containing the step number within the episode for each time step.
        :param evaluation_mode:         Whether to run the model in evaluation mode. If None, the model's default will
                                        be used.
        :param norm_mask:               An optional B x S tensor that is true for time steps that should be considered
                                        for updating normalization statistics and false for time steps that should be
                                        ignored.
        :param max_output_step_count:   An optional integer that limits the number of output steps. If not None, the
                                        agent will only produce outputs for the last max_output_step_count time steps.
        :return: A tuple consisting of the values of the given sequence (B x S) and an action distribution (B x S x A).
        """
        pass

    @abstractmethod
    def init_memory(self, batch_shape: tuple[int, ...]) -> AgentMemoryType:
        """
        Initializes the memory state of the agent.
        :param batch_size:          The batch size to initialize the memory state for.
        :return: The initialized memory state.
        """
        pass
