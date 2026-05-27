from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeVar, Generic

import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax
from flax.core.scope import VariableDict

from agent import (
    ValueFunctionCriticHead,
    PredictorHeadTree,
    BaseAgent,
    ActorHeadTree,
    CriticHead,
    CompositeAgent,
    ActorCriticAgentMemoryState,
    ActorCriticAgentStructure,
    ActorCriticAgentOutput,
)
from ap_gym import BaseActivePerceptionVectorEnv
from jax_util import tree_freeze, tree_sum
from models import MultiModalSequenceEncoder
from trajectory_buffer import TrajectoryBuffer
from .base_algorithm import (
    BaseAlgorithm,
    AlgorithmSettings,
    TrainingState,
    Trajectory,
    EnvironmentState,
    BaseAlgorithmConfig,
)
from .modules import OptimizerConfig, AgentConfig

ActorMemoryStateType = TypeVar("ActorMemoryStateType")
PredictorMemoryStateType = TypeVar("PredictorMemoryStateType")
CriticMemoryStateType = TypeVar("CriticMemoryStateType")

HAMAgentMemoryStateType = ActorCriticAgentMemoryState[
    ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType
]


class HAMAgent(
    CompositeAgent[ActorCriticAgentStructure],
    BaseAgent[ActorCriticAgentOutput, HAMAgentMemoryStateType],
    Generic[ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType],
):
    heads: ActorCriticAgentStructure[ActorHeadTree, PredictorHeadTree, CriticHead]
    sequence_encoders: ActorCriticAgentStructure[
        MultiModalSequenceEncoder, MultiModalSequenceEncoder, MultiModalSequenceEncoder
    ]


@flax.struct.dataclass
class HAMState:
    optimizer_state: optax.OptState


@dataclass
class HAMConfig(BaseAlgorithmConfig):
    # The discount factor
    gamma: float

    # The coefficient weighting between policy loss and prediction loss
    beta: float

    # The coefficient weighting the value loss
    value_loss_coeff: float

    # HAM takes one gradient update for each batch_size episodes
    batch_size: int

    # The optimizer configuration
    optimizer: OptimizerConfig

    # The agent configuration
    agent: AgentConfig

    _type: str = "HAM"  # Do not override

    def make_algorithm(self) -> "HAM":
        return HAM(self)


class HAM(
    BaseAlgorithm[HAMAgent, ActorCriticAgentOutput, HAMState, Trajectory, HAMConfig]
):
    """
    Haptic Attention Model (HAM) algorithm.
    Fleer, Sascha, et al. "Learning efficient haptic shape exploration with a rigid tactile sensor array."
    PloS one 15.1 (2020): e0226880.
    Code: https://github.com/fleer/Haptic-Attention-Model
    """

    @flax.struct.dataclass
    class _SubStepState:
        rng: jax.Array
        optimizer_state: optax.OptState
        agent_state: Any
        metrics: dict[str, jax.Array]
        metrics_step_1: dict[str, jax.Array]
        effective_steps: int

    def __init__(self, config: HAMConfig):
        super().__init__("ham", config=config, on_policy=True)
        self.__optimizer = None

    def __get_initial_algorithm_state(self, initial_agent_state: Any) -> HAMState:
        return HAMState(self.__optimizer.init(initial_agent_state["params"]))

    def _get_initial_algorithm_state(self, initial_agent_state: Any) -> HAMState:
        if (
            self.memory_horizon_with_context
            > self.algorithm_settings.batch_env_steps_per_iteration
        ):
            raise ValueError(
                "The memory horizon with context must be smaller or equal to the batch environment steps collected per"
                "iteration."
            )
        self.__optimizer = self.config.optimizer.make_optimizer(self.total_iterations)
        return self.__get_initial_algorithm_state(initial_agent_state)

    def _mk_agent(self, vector_env: BaseActivePerceptionVectorEnv) -> HAMAgent:
        actor_head = self.config.agent.make_actor_head(vector_env)
        predictor_head = self.config.agent.make_predictor_head(vector_env)
        init = (
            nn.initializers.orthogonal(1.0)
            if self.config.agent.orthogonal_layer_init
            else nn.linear.default_kernel_init
        )
        critic_head = ValueFunctionCriticHead(kernel_init=init)
        heads = ActorCriticAgentStructure(
            actor=actor_head, predictor=predictor_head, critic=critic_head
        )
        sequence_encoders = ActorCriticAgentStructure(
            *self.config.agent.make_sequence_encoders(
                vector_env.single_action_space.inner_action_space,
                vector_env.single_observation_space,
                self.max_episode_steps,
                count=3,
                max_episode_steps=self.max_episode_steps,
                network_probe_level=self.config.metric_log_level,
            )
        )

        return HAMAgent(
            sequence_encoders=sequence_encoders,
            heads=heads,
            observe_actions=self.config.agent.observe_actions,
        )

    def _get_algorithm_settings(self) -> AlgorithmSettings:
        if self.config.batch_size % self.train_env.num_envs != 0:
            raise ValueError(
                f"The batch size ({self.config.batch_size}) must be a multiple of the number of environments "
                f"({self.train_env.num_envs})."
            )
        num_episodes_per_env = self.config.batch_size // self.train_env.num_envs
        batch_env_steps_per_iteration = self.max_episode_steps * num_episodes_per_env
        return AlgorithmSettings(
            trajectory_buffer_size=batch_env_steps_per_iteration,
            batch_env_steps_per_iteration=batch_env_steps_per_iteration,
        )

    def compute_reward_to_go(
        self,
        rewards: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        next_terminated: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        CarryType = tuple[jax.Array, jax.Array, jax.Array]

        def scan_func(
            carry: CarryType, args: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[CarryType, tuple[jax.Array, jax.Array]]:
            next_reward_to_go, next_valid, next_terminated = carry
            reward, terminated, truncated = args
            valid = ~truncated & (terminated | next_valid)
            reward_to_go = (
                reward + self.config.gamma * next_reward_to_go * ~next_terminated
            ) * valid
            return (reward_to_go, valid, terminated), (
                reward_to_go,
                valid & ~terminated,
            )

        _, (reward_to_go, valid) = jax.lax.scan(
            scan_func,
            (jnp.zeros_like(rewards[:, -1]), next_terminated, next_terminated),
            (rewards.T, terminated.T, truncated.T),
            reverse=True,
        )

        return reward_to_go.T, valid.T

    def _handle_model_reset(
        self, iteration: jax.Array, algorithm_state: HAMState, agent_state: Any
    ) -> HAMState:
        return self.__get_initial_algorithm_state(agent_state)

    def _training_step(
        self,
        iteration: jax.Array,
        rng: jax.Array,
        training_state: TrainingState[HAMState],
        trajectory_buffer: TrajectoryBuffer[Trajectory],
        agent_memory_state: Any,
        current_env_state: EnvironmentState,
    ) -> tuple[TrainingState[HAMState], dict[str, Any]]:
        # Optimizing the policy and value network
        optimizer_state = training_state.algorithm_state.optimizer_state
        agent_state = training_state.agent_state

        # The advantage estimation will fail if the replay buffer is not full
        assert (
            trajectory_buffer.capacity_steps
            == self.algorithm_settings.batch_env_steps_per_iteration
        )
        data = trajectory_buffer.data_aligned

        rng, agent_rng = jax.random.split(rng)

        reward_to_go, data_valid = self.compute_reward_to_go(
            data.variables.reward,
            data.variables.terminated,
            data.variables.truncated,
            current_env_state.terminated,
        )

        def loss_fn(params: VariableDict):
            variables = flax.core.copy(agent_state, add_or_replace={"params": params})
            (agent_output, _), state_updates = self.agent.apply(
                variables,
                data.variables.obs,
                data.variables.prev_action,
                data.start,
                step_no=data.step_no,
                rngs=agent_rng,
                mutable=["batch_stats"],
                evaluation_mode=False,
            )
            log_prob = tree_sum(
                agent_output.action_distr.log_prob(tree_freeze(data.variables.action)),
                agent_output.critic.shape,
            )

            critic_error = reward_to_go - agent_output.critic

            # Policy loss
            # We deviate from the original paper here and use the reward-to-go to compute the baseline and not the
            # immediate reward.
            pg_loss_per_step = -jax.lax.stop_gradient(critic_error) * log_prob
            pg_loss = pg_loss_per_step.mean(where=data_valid)

            # Value loss
            v_loss = (critic_error**2).mean(where=data_valid)

            prediction_loss = self.train_env.loss_fn.jax(
                agent_output.prediction,
                data.variables.next_metadata.perception_target,
                agent_output.critic.shape,
            ).mean(where=data_valid)

            metrics = {"pg_loss": pg_loss, "v_loss": v_loss, "p_loss": prediction_loss}
            loss = (
                self.config.beta * pg_loss
                + prediction_loss
                + self.config.value_loss_coeff * v_loss
            )
            return loss, (metrics, state_updates)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, state_updates)), grads = grad_fn(agent_state["params"])
        new_params, new_opt_state = self.__optimizer.update(
            grads, optimizer_state, agent_state["params"]
        )
        new_agent_state = flax.core.copy(
            agent_state, add_or_replace={"params": new_params, **state_updates}
        )

        inner_optimizer_state = jax.tree.flatten(
            optimizer_state,
            is_leaf=lambda n: isinstance(n, optax.InjectStatefulHyperparamsState),
        )[0][0]

        # The above as dict
        metrics = {
            "ham": {
                "policy_loss": metrics["pg_loss"],
                "value_loss": metrics["v_loss"],
                "prediction_loss": metrics["p_loss"],
            },
            "general": {
                "learning_rate": inner_optimizer_state.hyperparams["learning_rate"],
            },
        }

        training_state = training_state.replace(
            agent_state=new_agent_state, algorithm_state=HAMState(new_opt_state)
        )
        return training_state, metrics
