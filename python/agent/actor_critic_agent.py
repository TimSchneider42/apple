from __future__ import annotations

from typing import TypeVar, Generic

import flax.struct
import jax

from distribution_tree import DistributionTree
from pytree import PyTree
from .base_agent import BaseAgentOutput

ActorStructureType = TypeVar("ActorStructureType")
PredictorStructureType = TypeVar("PredictorStructureType")
CriticStructureType = TypeVar("CriticStructureType")


@flax.struct.dataclass
class ActorCriticAgentStructure(
    Generic[ActorStructureType, PredictorStructureType, CriticStructureType],
    BaseAgentOutput,
):
    actor: ActorStructureType
    predictor: PredictorStructureType
    critic: CriticStructureType

    @property
    def action_distr(self) -> ActorStructureType:
        return self.actor

    @property
    def prediction(self) -> PredictorStructureType:
        return self.predictor


ActorMemoryStateType = TypeVar("ActorMemoryStateType")
PredictorMemoryStateType = TypeVar("PredictorMemoryStateType")
CriticMemoryStateType = TypeVar("CriticMemoryStateType")


@flax.struct.dataclass
class ActorCriticAgentMemoryState(
    ActorCriticAgentStructure[
        ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType
    ],
    Generic[ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType],
):
    pass


ActorCriticAgentOutput = ActorCriticAgentStructure[
    DistributionTree, PyTree[jax.Array], jax.Array
]
