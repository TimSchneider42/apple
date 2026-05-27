from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Generic, TypeVar

import distrax
import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from ap_gym import BaseActivePerceptionVectorEnv
from flax.core.scope import VariableDict

from agent import (
    QFunctionCriticEnsembleHead,
    ActorCriticAgentMemoryState,
    ActorCriticAgentOutput,
    BaseAgent,
    CompositeAgent,
    ActorCriticAgentStructure,
    PredictorHeadTree,
    ActorHeadTree,
    DenseQFunctionBackbone,
)
from distribution_tree import DistributionTree
from gym_space_map import gym_space_map
from jax_util import (
    tree_sum,
    get_act_fn,
    dict_mask,
    RecursiveMapping,
    dict_update,
    tree_unfreeze,
    metrics_set_valid_flag,
)
from metric_log_level import MetricLogLevel
from models import (
    MultiModalSequenceEncoder,
    MultiModalSequenceEmbedding,
    NORMALIZATION_FACTORIES,
    Normalization,
)
from pytree import PyTree
from trajectory_buffer import TrajectoryBuffer, TrajectoryBufferData
from .base_algorithm import (
    BaseAlgorithm,
    AlgorithmSettings,
    TrainingState,
    Trajectory,
    EnvironmentState,
    BaseAlgorithmConfig,
)
from .gym_space_embedding import embeddings_from_space
from .modules import (
    OptimizerConfig,
    AgentConfig,
    NormalizedScheduleWrapperState,
    ScheduleConfig,
    NormalizedScheduleWrapper,
    extract_optimizer_metrics,
    ParameterTransformation,
)
from .wrappers import Discrete32

logger = logging.getLogger(__name__)

ActorMemoryStateType = TypeVar("ActorMemoryStateType")
PredictorMemoryStateType = TypeVar("PredictorMemoryStateType")
CriticMemoryStateType = TypeVar("CriticMemoryStateType")

SACAgentMemoryStateType = ActorCriticAgentMemoryState[
    ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType
]


class DiscreteTargetEntropyMode(str, Enum):
    DEFAULT = "default"
    EPS_GREEDY_BASED = "eps_greedy_based"


@flax.struct.dataclass
class ValueEstimateOutput:
    value: jax.Array
    action_entropy: jax.Array


class SACAgent(
    BaseAgent[ActorCriticAgentOutput, SACAgentMemoryStateType],
    Generic[ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType],
):
    sequence_encoders: ActorCriticAgentStructure[
        MultiModalSequenceEncoder, MultiModalSequenceEncoder, MultiModalSequenceEncoder
    ]
    actor_head: ActorHeadTree
    predictor_head: PredictorHeadTree
    q_function: QFunctionCriticEnsembleHead
    observe_actions: bool = False
    hint_sample: PyTree[jax.Array] = None  # Only used during initialization

    def setup(self):
        critic_head = lambda x: x
        heads = ActorCriticAgentStructure(
            self.actor_head, self.predictor_head, critic_head
        )
        self.composite_agent = CompositeAgent(
            sequence_encoders=self.sequence_encoders,
            heads=heads,
            observe_actions=self.observe_actions,
        )

    def __call__(
        self,
        observations: PyTree[jax.Array],
        actions: PyTree[jax.Array],
        sequence_starts: jax.Array,
        memory_state: SACAgentMemoryStateType | None = None,
        step_no: jax.Array | None = None,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
        max_output_step_count: int | None = None,
    ) -> tuple[ActorCriticAgentOutput, SACAgentMemoryStateType]:
        agent_output, new_mem_state = self.composite_agent(
            observations,
            actions,
            sequence_starts,
            memory_state,
            step_no=step_no,
            evaluation_mode=evaluation_mode,
            norm_mask=norm_mask,
            max_output_step_count=max_output_step_count,
        )

        # Have to call the Q-function during initialization to make sure the parameters are initialized
        if self.is_initializing():
            self.q_function(
                agent_output.critic,
                agent_output.action_distr.sample(seed=jax.random.PRNGKey(0)),
                hint=jax.tree.map(
                    lambda x: jnp.repeat(x[None], agent_output.critic.shape[0], axis=0)[
                        :, None
                    ],
                    self.hint_sample,
                ),
                evaluation_mode=evaluation_mode,
                norm_mask=norm_mask,
            )

        return agent_output, new_mem_state

    def init_memory(self, batch_shape: tuple[int, ...]) -> SACAgentMemoryStateType:
        return self.composite_agent.init_memory(batch_shape)

    def estimated_expected_value(
        self,
        critic_hidden_state: jax.Array,
        action_distr: DistributionTree,
        *,
        hint: PyTree[jax.Array] = None,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> ValueEstimateOutput:
        actions, action_log_probs = action_distr.sample_and_log_prob(
            seed=self.make_rng("action")
        )
        conditional_value = self.q_function(
            critic_hidden_state,
            actions,
            hint=hint,
            evaluation_mode=evaluation_mode,
            norm_mask=norm_mask,
        )
        return self.take_expectation(conditional_value, action_distr, action_log_probs)

    def estimate_value(
        self,
        critic_hidden_state: jax.Array,
        actions: PyTree[jax.Array],
        *,
        hint: PyTree[jax.Array] = None,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        conditional_value = self.q_function(
            critic_hidden_state,
            actions,
            hint=hint,
            evaluation_mode=evaluation_mode,
            norm_mask=norm_mask,
        )
        return self.select_value(conditional_value, actions)

    def combined_q_evaluation(
        self,
        critic_hidden_state: jax.Array,
        next_action_distr_target: DistributionTree,
        actions: PyTree[jax.Array],
        *,
        hint: PyTree[jax.Array] = None,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, ValueEstimateOutput]:
        (
            next_actions_target,
            next_action_log_probs_target,
        ) = next_action_distr_target.sample_and_log_prob(seed=self.make_rng("action"))
        combined_next_actions = jax.tree.map(
            lambda a, b: jnp.concatenate([a, b], axis=0),
            actions,
            tree_unfreeze(next_actions_target),
        )
        combined_hidden_state = jnp.tile(critic_hidden_state, (2, 1, 1))
        combined_hint = jax.tree.map(
            lambda x: jnp.tile(x, (2,) + (1,) * (len(x.shape) - 1)), hint
        )
        combined_norm_mask = jnp.tile(norm_mask, (2, 1))
        combined_conditional_value = self.q_function(
            combined_hidden_state,
            combined_next_actions,
            hint=combined_hint,
            evaluation_mode=evaluation_mode,
            norm_mask=combined_norm_mask,
        )
        conditional_value, conditional_value_target = jnp.split(
            combined_conditional_value, 2, axis=1
        )
        value = self.select_value(conditional_value, actions)
        value_target = self.take_expectation(
            conditional_value_target,
            next_action_distr_target,
            next_action_log_probs_target,
        )
        return value, value_target

    def take_expectation(
        self,
        conditional_value: jax.Array,
        next_action_distr: DistributionTree,
        next_action_log_probs: PyTree[jax.Array],
    ) -> ValueEstimateOutput:
        # We use two different methods here to deal with continuous and discrete actions. Continuous actions simply
        # become inputs to the Q-function, while discrete actions are used to index into the Q-function output.
        batch_shape = conditional_value.shape[
            1 : len(conditional_value.shape) - self.action_leaf_count
        ]
        discrete_log_probs = jax.tree.map(
            lambda d: d.logits if hasattr(d, "logits") else jnp.zeros(batch_shape),
            next_action_distr.orig_tree,
            is_leaf=lambda d: isinstance(d, distrax.Distribution),
        )
        discrete_log_probs_arr = jax.tree.leaves(discrete_log_probs)
        discrete_log_probs_arr_reshaped = [
            d.reshape(
                batch_shape
                + (1,) * i
                + (-1,)
                + (1,) * (len(discrete_log_probs_arr) - 1 - i)
            )
            for i, d in enumerate(discrete_log_probs_arr)
        ]

        if len(discrete_log_probs_arr_reshaped) == 0:
            discrete_log_probs_arr_reshaped = [jnp.zeros(batch_shape)]

        discrete_log_probs_arr_joined: jax.Array = sum(discrete_log_probs_arr_reshaped)
        continuous_log_probs_tree = jax.tree.map(
            lambda log_prob, d: (
                jnp.zeros(batch_shape) if hasattr(d, "logits") else log_prob
            ),
            next_action_log_probs,
            next_action_distr.orig_tree,
        )
        continuous_log_probs = tree_sum(continuous_log_probs_tree, batch_shape)
        non_batch_dimensions = range(
            len(batch_shape), len(discrete_log_probs_arr_joined.shape)
        )
        action_log_probs = discrete_log_probs_arr_joined + continuous_log_probs.reshape(
            batch_shape + (1,) * len(non_batch_dimensions)
        )
        discrete_action_probs = jnp.exp(discrete_log_probs_arr_joined)
        value = jnp.sum(
            discrete_action_probs * conditional_value,
            axis=tuple(i + 1 for i in non_batch_dimensions),
        )
        action_entropy = -jnp.sum(
            discrete_action_probs * action_log_probs, axis=non_batch_dimensions
        )
        return ValueEstimateOutput(value, action_entropy)

    def select_value(
        self, conditional_value: jax.Array, next_actions: PyTree[jax.Array]
    ) -> jax.Array:
        batch_shape = conditional_value.shape[
            1 : len(conditional_value.shape) - self.action_leaf_count
        ]
        discrete_actions_arr = jax.tree.map(
            lambda act, ah: (
                jnp.zeros(batch_shape, dtype=jnp.int32)
                if jnp.issubdtype(act.dtype, jnp.floating)
                else act
            ),
            next_actions,
            tree_unfreeze(self.actor_head.orig_tree),
        )
        discrete_actions_arr_lst = jax.tree.leaves(discrete_actions_arr)
        if len(discrete_actions_arr_lst) == 0:
            discrete_actions_arr_joined = ()
        else:
            discrete_actions_arr_joined = tuple(
                jnp.stack(discrete_actions_arr_lst, axis=0)
            )
        batch_shape_indices = tuple(
            jnp.arange(d).reshape((1,) * i + (-1,) + (1,) * (len(batch_shape) - 1 - i))
            for i, d in enumerate(batch_shape)
        )
        value = conditional_value[
            (slice(None, None, None),)
            + tuple(batch_shape_indices)
            + tuple(discrete_actions_arr_joined)
        ]
        return value

    def get_effective_sequence_encoders(self):
        return self.composite_agent.get_effective_sequence_encoders()

    @property
    def action_leaf_count(self):
        return len(jax.tree.flatten(self.actor_head.orig_tree)[0])


@flax.struct.dataclass
class SACState:
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    alpha_optimizer_state: optax.OptState
    target_agent_state: Any
    log_alpha: jax.Array
    gradient_step: jax.Array
    target_entropy_scale_discrete_schedule_state: NormalizedScheduleWrapperState
    target_entropy_discrete_epsilon_schedule_state: NormalizedScheduleWrapperState
    target_entropy_scale_continuous_schedule_state: NormalizedScheduleWrapperState


class CriticBackboneType(str, Enum):
    DENSE = "dense"
    TRANSFORMER_MLP_BLOCK = "transformer_mlp_block"


@dataclass
class SACConfig(BaseAlgorithmConfig):
    # The ratio of update steps to environment steps
    update_to_data_ratio: float

    # The initial ratio of update steps to environment steps. Linearly annealed to target during UTD warmup steps
    initial_utd: float | None

    # The relative number of steps to linearly anneal the UTD ratio to the target value
    utd_warmup_steps_rel: float

    # Number of samples to evaluate in parallel during training
    batch_size: int

    # The size of the replay buffer
    buffer_size: int

    # The discount factor gamma
    gamma: float

    # Target smoothing coefficient
    tau: float

    # The interval of training policy (delayed)
    policy_interval: int

    # For how many steps to train the policy on the same batch
    policy_update_steps: int

    # The interval of updates for the target networks
    target_network_interval: int

    # Noise clip parameter of the Target Policy Smoothing Regularization
    noise_clip: float

    # Entropy regularization coefficient
    alpha: float

    # Automatic tuning of the entropy coefficient
    autotune: bool

    # Coefficient for scaling the autotune entropy target for discrete actions
    target_entropy_scale_discrete: ScheduleConfig

    # Coefficient for scaling the autotune entropy target for continuous actions
    target_entropy_scale_continuous: ScheduleConfig

    # Embedding dimension of the action and hint of the Q-function (None means no embedding)
    critic_action_hint_embedding_dims: int | None

    # Embedding dimension of the state of the Q-function (None means no embedding)
    critic_state_embedding_dims: int | None

    # Hidden dimensions of the Q-function (None means take the sum of action-hint and state embedding dimensions
    critic_hidden_dims: int | None

    # Number of hidden layers in the Q function
    critic_hidden_layers: int

    # Activation function for the Q function
    critic_activation: str

    # Which norm to use in the critic
    critic_norm: Normalization

    # Which norm to use for the action-hint inputs of the critic
    critic_action_hint_norm: Normalization

    # Which norm to use for the state input of the critic
    critic_state_norm: Normalization

    # Which norm to use for the state/action-hint encodings of the critic
    critic_post_enc_norm: Normalization

    # Use skip connections in the Q function
    critic_skip_connections: bool

    # Coefficient of the regular actor loss
    a_coef: float

    # Coefficient of the prediction loss
    p_coef: float

    # Use target networks for the Q function
    use_target_networks: bool

    # Compute targets in train mode instead of eval. Faster but leads to different targets with batch norm
    compute_targets_in_train_mode: bool

    # Provide the prediction target as a hint to the critic
    use_critic_hints: bool

    # Use target networks for the sequence encoders
    encoder_use_target_networks: bool

    # Disable the critic update
    disable_critic_update: bool

    # The number of batched environment steps per iteration
    batch_env_steps_per_iteration: int

    # The optimizer configuration for the actor
    actor_optimizer: OptimizerConfig

    # The optimizer configuration for the critic
    critic_optimizer: OptimizerConfig

    # The optimizer configuration for the alpha parameter
    alpha_optimizer: OptimizerConfig

    # The agent configuration
    agent: AgentConfig

    # The number of Q functions in the ensemble
    q_ensemble_size: int

    # Interval (in number of actor update steps) at which to update the alpha parameter
    alpha_update_interval: int

    # Whether to keep the critic in training mode during the actor update
    critic_use_training_mode_during_actor_update: bool

    # How to compute the discrete target entropy. Default is scale * ln(|A|), hence scale share of the maximum possible
    # entropy. Epsilon greedy based computes the target entropy such that the stochastic policy is at least as
    # stochastic as an epsilon-greedy policy.
    # See https://discuss.ray.io/t/target-entropy-in-discrete-sac-implementation/12182
    discrete_target_entropy_mode: DiscreteTargetEntropyMode

    # Epsilon for the epsilon greedy based target entropy mode.
    eps_greedy_target_entropy_mode_epsilon: ScheduleConfig

    _type: str = "SAC"  # Do not override

    def make_algorithm(self) -> "SAC":
        return SAC(self)


class SAC(
    BaseAlgorithm[SACAgent, ActorCriticAgentOutput, SACState, Trajectory, SACConfig]
):
    @flax.struct.dataclass
    class _SubStepState:
        rng: jax.Array
        actor_optimizer_state: optax.OptState
        critic_optimizer_state: optax.OptState
        alpha_optimizer_state: optax.OptState
        agent_state: Any
        target_agent_state: Any
        metrics: dict[str, jax.Array]
        metrics_step_1: dict[str, jax.Array]
        log_alpha: jax.Array

    def __init__(self, config: SACConfig):
        super().__init__("sac", config=config, on_policy=False)
        self.__actor_optimizer = self.__critic_optimizer = self.__alpha_optimizer = None
        self.__param_mask_actor = self.__param_mask_critic = None
        self.__target_entropy_scale_discrete_schedule: (
            NormalizedScheduleWrapper | None
        ) = None
        self.__target_entropy_scale_continuous_schedule: (
            NormalizedScheduleWrapper | None
        ) = None
        self.__empty_metrics: dict[str, Any] | None = None

    def _process_initial_agent_state(
        self, initial_agent_state: dict[str, Any]
    ) -> dict[str, Any]:
        # The point of this is to execute the weight normalization once in the beginning, so that we do not log any
        # large weights in the first step.
        def make_single_step(
            optimizer: optax.GradientTransformation, params: Any
        ) -> dict[str, Any]:
            new_params, _ = optimizer.update(
                jax.tree.map(jnp.zeros_like, params),
                optimizer.init(params),
                params,
            )
            return new_params

        params = initial_agent_state["params"]
        params = make_single_step(
            self.config.critic_optimizer.make_optimizer(0.0), params
        )
        params = make_single_step(
            self.config.actor_optimizer.make_optimizer(0.0), params
        )

        output = initial_agent_state.copy()
        output["params"] = params

        return output

    def __get_initial_algorithm_state(
        self,
        initial_agent_state: Any,
        current_iteration: jax.Array,
        target_entropy_scale_discrete_schedule_state: NormalizedScheduleWrapperState,
        target_entropy_discrete_epsilon_schedule_state: NormalizedScheduleWrapperState,
        target_entropy_scale_continuous_schedule_state: NormalizedScheduleWrapperState,
        initial_gradient_step: jax.Array = jnp.array(0),
    ) -> SACState:
        if self.config.autotune:
            initial_log_alpha = jnp.zeros(())
        else:
            initial_log_alpha = jnp.log(self.config.alpha)

        training_segment = current_iteration // self.model_reset_interval_iterations
        training_segment_start = training_segment * self.model_reset_interval_iterations
        training_segment_end = (
            training_segment_start + self.model_reset_interval_iterations
        )
        training_segment_updates = self.__get_total_updates_at_itr(
            training_segment_end
        ) - self.__get_total_updates_at_itr(training_segment_start)

        actor_optimizer_state = self.__actor_optimizer.init(
            dict_mask(initial_agent_state["params"], self.__param_mask_actor)
        )
        critic_optimizer_state = self.__critic_optimizer.init(
            dict_mask(initial_agent_state["params"], self.__param_mask_critic)
        )
        alpha_optimizer_state = self.__alpha_optimizer.init(initial_log_alpha)

        self.__empty_metrics = self.__get_empty_metrics(
            actor_optimizer_state, critic_optimizer_state
        )

        def set_optimizer_step_limit(optimizer_state: Any, limit: jax.Array) -> Any:
            return jax.tree.map(
                lambda s: (
                    s.replace(limit=limit)
                    if isinstance(s, NormalizedScheduleWrapperState)
                    else s
                ),
                optimizer_state,
                is_leaf=lambda s: isinstance(s, NormalizedScheduleWrapperState),
            )

        return SACState(
            set_optimizer_step_limit(actor_optimizer_state, training_segment_updates),
            set_optimizer_step_limit(critic_optimizer_state, training_segment_updates),
            set_optimizer_step_limit(alpha_optimizer_state, training_segment_updates),
            initial_agent_state if self.config.use_target_networks else None,
            initial_log_alpha,
            initial_gradient_step,
            target_entropy_scale_discrete_schedule_state,
            target_entropy_discrete_epsilon_schedule_state,
            target_entropy_scale_continuous_schedule_state,
        )

    def _get_initial_algorithm_state(self, initial_agent_state: Any) -> SACState:
        self.__actor_optimizer = self.config.actor_optimizer.make_optimizer(
            0.0,
            track_metrics=self.config.metric_log_level >= MetricLogLevel.VERY_DETAILED,
        )
        self.__critic_optimizer = self.config.critic_optimizer.make_optimizer(
            0.0,
            track_metrics=self.config.metric_log_level >= MetricLogLevel.VERY_DETAILED,
        )
        self.__alpha_optimizer = self.config.alpha_optimizer.make_optimizer(
            0.0,
            track_metrics=self.config.metric_log_level >= MetricLogLevel.VERY_DETAILED,
        )
        effective_seq_enc = self.agent.apply(
            initial_agent_state, method=self.agent.get_effective_sequence_encoders
        )

        self.__param_mask_actor = jax.tree.map(
            lambda _: True, initial_agent_state["params"]
        )
        self.__param_mask_actor["q_function"] = False
        self.__param_mask_actor["critic_head"] = False
        # Does any actor module share the sequence encoder with the critic?
        if (
            not effective_seq_enc.actor.critic
            and not effective_seq_enc.predictor.critic
        ):
            self.__param_mask_actor["sequence_encoders_critic"] = False

        self.__param_mask_critic = jax.tree.map(
            lambda _: True, initial_agent_state["params"]
        )
        self.__param_mask_critic["actor_head"] = False
        self.__param_mask_critic["predictor_head"] = False
        # Does the critic share the sequence encoder with the actor?
        if not effective_seq_enc.critic.actor:
            self.__param_mask_critic["sequence_encoders_actor"] = False
        # Does the critic share the sequence encoder with the predictor?
        if not effective_seq_enc.critic.predictor:
            self.__param_mask_critic["sequence_encoders_predictor"] = False

        self.__target_entropy_scale_discrete_schedule = (
            self.config.target_entropy_scale_discrete.make_schedule(
                self.total_iterations
            )
        )
        self.__target_entropy_discrete_epsilon_schedule = (
            self.config.eps_greedy_target_entropy_mode_epsilon.make_schedule(
                self.total_iterations
            )
        )
        self.__target_entropy_scale_continuous_schedule = (
            self.config.target_entropy_scale_continuous.make_schedule(
                self.total_iterations
            )
        )

        return self.__get_initial_algorithm_state(
            initial_agent_state,
            jnp.array(0),
            self.__target_entropy_scale_discrete_schedule.init(),
            self.__target_entropy_discrete_epsilon_schedule.init(),
            self.__target_entropy_scale_continuous_schedule.init(),
        )

    def _mk_agent(self, vector_env: BaseActivePerceptionVectorEnv) -> SACAgent:
        actor_head = self.config.agent.make_actor_head(vector_env)
        predictor_head = self.config.agent.make_predictor_head(vector_env)
        if self.config.agent.orthogonal_layer_init:
            logger.warning("Orthogonal layer init configuration ignored in SAC.")
        discrete_action_counts_flat = jax.tree.leaves(
            self.__get_discrete_state_counts(vector_env.single_inner_action_space)
        )
        embedding_factories_action = dict(
            self.config.agent.get_embedding_factories(
                network_probe_level=self.config.metric_log_level
            )
        )
        # Do not encode discrete actions, as they will be handled differently
        embedding_factories_action[Discrete32] = (
            lambda space: lambda x, evaluation_mode, norm_mask: jnp.empty(
                x.shape + (0,)
            )
        )
        if self.config.use_critic_hints:
            hint_embedding = MultiModalSequenceEmbedding(
                embeddings_from_space(
                    vector_env.single_prediction_target_space,
                    self.config.agent.get_embedding_factories(
                        network_probe_level=self.config.metric_log_level
                    ),
                )
            )
            hint_sample = vector_env.single_prediction_target_space.sample()
        else:
            hint_embedding = hint_sample = None

        backbone = DenseQFunctionBackbone(
            hidden_dims=self.config.critic_hidden_dims,
            hidden_layers=self.config.critic_hidden_layers,
            act_fn=get_act_fn(self.config.critic_activation),
            normalization=NORMALIZATION_FACTORIES[self.config.critic_norm],
            skip_connections=self.config.critic_skip_connections,
        )

        q_function = QFunctionCriticEnsembleHead(
            action_embedding=MultiModalSequenceEmbedding(
                embeddings_from_space(
                    vector_env.single_inner_action_space, embedding_factories_action
                )
            ),
            action_hint_embedding_dims=self.config.critic_action_hint_embedding_dims,
            state_embedding_dims=self.config.critic_state_embedding_dims,
            action_hint_normalization=NORMALIZATION_FACTORIES[
                self.config.critic_action_hint_norm
            ],
            state_normalization=NORMALIZATION_FACTORIES[self.config.critic_state_norm],
            post_encoding_norm=NORMALIZATION_FACTORIES[
                self.config.critic_post_enc_norm
            ],
            backbone=backbone,
            output_shape=tuple(discrete_action_counts_flat),
            hint_embedding=hint_embedding,
            ensemble_size=self.config.q_ensemble_size,
        )
        sequence_encoders = ActorCriticAgentStructure(
            *self.config.agent.make_sequence_encoders(
                vector_env.single_action_space.inner_action_space,
                vector_env.single_observation_space,
                memory_horizon=self.memory_horizon,
                count=3,
                max_episode_steps=self.max_episode_steps,
                network_probe_level=self.config.metric_log_level,
            )
        )

        return SACAgent(
            sequence_encoders=sequence_encoders,
            actor_head=actor_head,
            predictor_head=predictor_head,
            q_function=q_function,
            observe_actions=self.config.agent.observe_actions,
            hint_sample=hint_sample,
        )

    def _get_algorithm_settings(self) -> AlgorithmSettings:
        buffer_size = self.config.buffer_size
        max_buffer_size = self.config.total_env_steps // self.train_env.num_envs
        if buffer_size < 0:
            buffer_size = max_buffer_size
        else:
            buffer_size = min(buffer_size, max_buffer_size)
        logger.info(
            f"Determined optimal sample sequence length as {self.optimal_sample_sequence_length}."
        )
        return AlgorithmSettings(
            trajectory_buffer_size=buffer_size,
            batch_env_steps_per_iteration=self.config.batch_env_steps_per_iteration,
            min_learning_starts=self.optimal_sample_sequence_length
            * self.train_env.num_envs,
        )

    def _get_empty_metrics(self) -> dict[str, Any]:
        return self.__empty_metrics

    def __get_empty_metrics(
        self,
        initial_actor_optimizer_state: Any,
        initial_critic_optimizer_state: Any,
    ) -> dict[str, Any]:
        empty_actor_metrics = {
            "loss": np.nan,
            "prediction_loss": np.nan,
            "value_loss": np.nan,
            "entropy": np.nan,
            "alpha": np.nan,
            "optimizer": extract_optimizer_metrics(initial_actor_optimizer_state),
        }
        if self.config.autotune:
            empty_actor_metrics.update({"alpha_loss": np.nan, "target_entropy": np.nan})
        empty_critic_metrics = {
            **{f"qf{i}_values": np.nan for i in range(self.config.q_ensemble_size)},
            **{f"qft{i}_values": np.nan for i in range(self.config.q_ensemble_size)},
            **{f"qf{i}_loss": np.nan for i in range(self.config.q_ensemble_size)},
            "qf_loss": np.nan,
            "optimizer": extract_optimizer_metrics(initial_critic_optimizer_state),
        }

        empty_step_metrics = {
            "actor": empty_actor_metrics,
            "critic": empty_critic_metrics,
        }

        empty_metrics = {
            "sac": {
                **empty_step_metrics,
                "step1": empty_step_metrics,
            },
            "general": {
                "learning_rate_actor": np.nan,
                "learning_rate_critic": np.nan,
                "update_steps": -1,
                "total_update_steps": -1,
            },
        }
        return metrics_set_valid_flag(empty_metrics, True, zeros_like_fn=np.zeros_like)

    def _handle_model_reset(
        self, iteration: jax.Array, algorithm_state: SACState, agent_state: Any
    ) -> SACState:
        return self.__get_initial_algorithm_state(
            agent_state,
            iteration,
            initial_gradient_step=algorithm_state.gradient_step,
            target_entropy_scale_discrete_schedule_state=algorithm_state.target_entropy_scale_discrete_schedule_state,
            target_entropy_discrete_epsilon_schedule_state=algorithm_state.target_entropy_discrete_epsilon_schedule_state,
            target_entropy_scale_continuous_schedule_state=algorithm_state.target_entropy_scale_continuous_schedule_state,
        )

    def _training_step(
        self,
        iteration: jax.Array,
        rng: jax.Array,
        training_state: TrainingState[SACState],
        trajectory_buffer: TrajectoryBuffer[Trajectory],
        agent_memory_state: Any,
        current_env_state: EnvironmentState,
    ) -> tuple[TrainingState[SACState], dict[str, Any]]:
        context_length = self.memory_horizon_with_context - self.memory_horizon
        sample_sequence_length = self.optimal_sample_sequence_length - context_length
        trajectory_buffer = trajectory_buffer.populate_cache(
            sample_sequence_length,
        )

        def rem_ctx(tree: Any, axis: int = 1):
            return jax.tree.map(
                lambda x: x[
                    (slice(None),) * (axis % len(x.shape))
                    + (slice(context_length, None),)
                ],
                tree,
            )

        def loss_fn_critic(
            params: VariableDict,
            rng: jax.Array,
            batch_traj: TrajectoryBufferData,
            batch_traj_valid: jax.Array,
            agent_state: Any,
            target_agent_state: Any,
            log_alpha: jax.Array,
        ):
            batch_traj_current = jax.tree.map(lambda x: x[:, :-1], batch_traj)

            full_params = dict_update(agent_state["params"], params)
            variables = flax.core.copy(
                agent_state, add_or_replace={"params": full_params}
            )

            # Compute targets
            (
                rng_train,
                rng_eval,
                rng_target,
                rng_target_head,
                rng_q,
            ) = jax.random.split(rng, 5)

            (agent_output_train, _), state_updates = self.agent.apply(
                variables,
                batch_traj.variables.obs,
                batch_traj.variables.prev_action,
                batch_traj.start,
                step_no=batch_traj.step_no,
                rngs=rng_train,
                mutable=["batch_stats"],
                evaluation_mode=False,
                norm_mask=batch_traj_valid,
            )

            keep_state_updates = (
                not self.config.critic_use_training_mode_during_actor_update
                or not self.config.compute_targets_in_train_mode
            )

            if self.config.compute_targets_in_train_mode:
                agent_output_eval = agent_output_train
            else:
                (agent_output_eval, _), _ = self.agent.apply(
                    variables,
                    batch_traj.variables.obs,
                    batch_traj.variables.prev_action,
                    batch_traj.start,
                    step_no=batch_traj.step_no,
                    rngs=rng_eval,
                    mutable=[],
                    evaluation_mode=True,
                    norm_mask=batch_traj_valid,
                )
            agent_output_eval: ActorCriticAgentOutput

            if keep_state_updates:
                variables = flax.core.copy(variables, add_or_replace=state_updates)
            agent_output_train: ActorCriticAgentOutput

            if self.config.use_target_networks:
                if self.config.encoder_use_target_networks:
                    (agent_output_target, _), _ = self.agent.apply(
                        target_agent_state,
                        batch_traj.variables.obs,
                        batch_traj.variables.prev_action,
                        batch_traj.start,
                        step_no=batch_traj.step_no,
                        rngs=rng_target,
                        mutable=["batch_stats"],
                        evaluation_mode=not self.config.compute_targets_in_train_mode,
                        norm_mask=batch_traj_valid,
                    )
                else:
                    agent_output_target = agent_output_eval
                agent_output_target: ActorCriticAgentOutput

                # Note that the hint will be ignored if no hint embedder is configured
                target_value_estimate, _ = self.agent.apply(
                    target_agent_state,
                    agent_output_target.critic,
                    agent_output_eval.action_distr,
                    hint=batch_traj.variables.next_metadata.perception_target,
                    method=self.agent.estimated_expected_value,
                    rngs=rng_target_head,
                    mutable=["batch_stats"],
                    evaluation_mode=not self.config.compute_targets_in_train_mode,
                    norm_mask=batch_traj_valid,
                )
                target_value_estimate: ValueEstimateOutput

                q_values, state_updates = self.agent.apply(
                    variables,
                    agent_output_train.critic[:, :-1],
                    batch_traj_current.variables.action,
                    hint=batch_traj_current.variables.next_metadata.perception_target,
                    method=self.agent.estimate_value,
                    rngs=rng_q,
                    mutable=["batch_stats"],
                    evaluation_mode=False,
                    norm_mask=batch_traj_valid[:, :-1],
                )
            else:
                (
                    q_values,
                    target_value_estimate,
                ), state_updates = self.agent.apply(
                    variables,
                    agent_output_train.critic,
                    agent_output_eval.action_distr,
                    batch_traj.variables.action,
                    hint=batch_traj.variables.next_metadata.perception_target,
                    method=self.agent.combined_q_evaluation,
                    rngs=rng_target_head,
                    mutable=["batch_stats"],
                    evaluation_mode=False,
                    norm_mask=batch_traj_valid,
                )
                q_values = q_values[..., :-1]

            q_values = rem_ctx(q_values, axis=-1)
            terminated = rem_ctx(batch_traj.variables.terminated)
            truncated = rem_ctx(batch_traj.variables.truncated)
            non_terminal = ~(terminated | truncated)

            next_raw_value = rem_ctx(target_value_estimate.value, axis=-1)[..., 1:]
            next_entropy = rem_ctx(target_value_estimate.action_entropy)[..., 1:]

            next_value = next_raw_value.min(0) + jnp.exp(log_alpha) * next_entropy
            next_terminated = terminated[..., 1:]

            base_reward = rem_ctx(
                batch_traj_current.variables.next_metadata.base_reward
            )

            prediction_loss = self.train_env.loss_fn.jax(
                jax.tree.map(
                    lambda p: p[:, :-1],
                    rem_ctx(agent_output_eval.prediction),
                ),
                rem_ctx(batch_traj_current.variables.next_metadata.perception_target),
                base_reward.shape[:2],
            )

            reward = base_reward - prediction_loss

            # If the current state is terminal, then there cannot be any further action and, thus, the Q-value
            # is not defined.
            target_q_values_valid = non_terminal[..., :-1]
            target_q_values_valid_train = target_q_values_valid

            target_q_values = reward + ~next_terminated * self.config.gamma * next_value

            squared_diff = (q_values - jax.lax.stop_gradient(target_q_values)) ** 2
            qf_losses_training = squared_diff.mean(
                axis=(1, 2), where=target_q_values_valid_train
            )
            qf_loss_training = qf_losses_training.sum()

            has_training_data = target_q_values_valid_train.sum() > 0

            qf_loss_training = jax.lax.select(
                has_training_data, qf_loss_training, jnp.array(0.0)
            )

            metrics = {
                **{
                    f"qf{i}_values": q_values[i].mean()
                    for i in range(self.config.q_ensemble_size)
                },
                **{
                    f"_qf{i}_values": has_training_data
                    for i in range(self.config.q_ensemble_size)
                },
                **{
                    f"qft{i}_values": target_q_values[i].mean()
                    for i in range(self.config.q_ensemble_size)
                },
                **{
                    f"_qft{i}_values": has_training_data
                    for i in range(self.config.q_ensemble_size)
                },
                **{
                    f"qf{i}_loss": qf_losses_training[i]
                    for i in range(self.config.q_ensemble_size)
                },
                **{
                    f"_qf{i}_loss": has_training_data
                    for i in range(self.config.q_ensemble_size)
                },
                "qf_loss": qf_loss_training,
                "_qf_loss": has_training_data,
            }
            return jax.lax.select(has_training_data, qf_loss_training, 0.0), (
                metrics,
                state_updates if keep_state_updates else {},
            )

        def loss_fn_actor(
            params: VariableDict,
            rng: jax.Array,
            batch_traj: Any,
            batch_traj_valid: jax.Array,
            agent_state: Any,
            log_alpha: jax.Array,
        ):
            full_params = dict_update(agent_state["params"], params)
            variables = flax.core.copy(
                agent_state, add_or_replace={"params": full_params}
            )

            embed_rng, head_rng = jax.random.split(rng)
            (agent_output, _), state_updates = self.agent.apply(
                variables,
                batch_traj.variables.obs,
                batch_traj.variables.prev_action,
                batch_traj.start,
                step_no=batch_traj.step_no,
                rngs=embed_rng,
                mutable=["batch_stats"],
                evaluation_mode=False,
                norm_mask=batch_traj_valid,
            )
            agent_output: ActorCriticAgentOutput = agent_output

            value_estimate, _ = self.agent.apply(
                variables,
                jax.lax.stop_gradient(agent_output.critic),
                agent_output.action_distr,
                hint=batch_traj.variables.next_metadata.perception_target,
                method=self.agent.estimated_expected_value,
                rngs=head_rng,
                mutable=["batch_stats"],
                evaluation_mode=not self.config.critic_use_training_mode_during_actor_update,
                norm_mask=batch_traj_valid,
            )
            value_estimate: ValueEstimateOutput

            not_terminated = ~rem_ctx(batch_traj.variables.terminated)

            full_value_loss = -rem_ctx(value_estimate.value, axis=-1).min(0)
            full_action_entropy = rem_ctx(value_estimate.action_entropy)

            full_prediction_loss = self.train_env.loss_fn.jax(
                rem_ctx(agent_output.prediction),
                rem_ctx(batch_traj.variables.next_metadata.perception_target),
                (self.config.batch_size, sample_sequence_length),
            )

            full_actor_loss = full_value_loss - jnp.exp(log_alpha) * full_action_entropy
            full_actor_loss = (
                self.config.a_coef * full_actor_loss
                + self.config.p_coef * full_prediction_loss
            )

            def masked_loss(
                mask: jax.Array,
            ):
                has_data = mask.sum() > 0

                actor_loss_scalar = full_actor_loss.mean(where=mask)
                action_entropy_scalar = full_action_entropy.mean(where=mask)

                metrics = {
                    "loss": actor_loss_scalar,
                    "_loss": has_data,
                    "prediction_loss": full_prediction_loss.mean(where=mask),
                    "_prediction_loss": has_data,
                    "value_loss": full_value_loss.mean(where=mask),
                    "_value_loss": has_data,
                    "entropy": action_entropy_scalar,
                    "_entropy": has_data,
                }
                return (
                    actor_loss_scalar,
                    action_entropy_scalar,
                    metrics,
                )

            actor_loss_valid_train = not_terminated
            (
                actor_loss,
                action_entropy,
                metrics,
            ) = masked_loss(
                actor_loss_valid_train,
            )

            has_training_data = actor_loss_valid_train.sum() > 0
            actor_loss = jax.lax.select(has_training_data, actor_loss, jnp.array(0.0))
            action_entropy = jax.lax.select(
                has_training_data, action_entropy, jnp.array(0.0)
            )

            return jax.lax.select(has_training_data, actor_loss, 0.0), (
                {**metrics},
                state_updates,
                action_entropy,
            )

        def sub_step_fn(
            sub_step: jax.Array, state: SAC._SubStepState
        ) -> SAC._SubStepState:
            def optimizer_step(
                loss_fn: Callable[[VariableDict], tuple[jax.Array, tuple[Any, Any]]],
                optimizer: ParameterTransformation,
                optimizer_state: optax.OptState,
                agent_state: Any,
                param_mask: RecursiveMapping[bool],
            ):
                reduced_params = dict_mask(agent_state["params"], param_mask)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, (metrics, state_updates, *rest)), grads = grad_fn(reduced_params)

                new_reduced_params, new_opt_state = optimizer.update(
                    grads, optimizer_state, reduced_params
                )
                new_params = dict_update(agent_state["params"], new_reduced_params)
                new_agent_state = flax.core.copy(
                    agent_state, add_or_replace={"params": new_params, **state_updates}
                )

                metrics["optimizer"] = extract_optimizer_metrics(new_opt_state)

                return (
                    new_agent_state,
                    new_opt_state,
                    metrics_set_valid_flag(metrics),
                    *rest,
                )

            def critic_update(
                rng: jax.Array,
                agent_state: Any,
                optimizer_state: optax.OptState,
            ):

                return optimizer_step(
                    partial(
                        loss_fn_critic,
                        rng=rng,
                        batch_traj=batch_traj_buffer.data,
                        batch_traj_valid=batch_traj_buffer.data_valid,
                        agent_state=agent_state,
                        target_agent_state=state.target_agent_state,
                        log_alpha=state.log_alpha,
                    ),
                    self.__critic_optimizer,
                    optimizer_state,
                    agent_state,
                    self.__param_mask_critic,
                )

            def actor_update(
                rng: jax.Array,
                agent_state: Any,
                optimizer_state: optax.OptState,
                log_alpha: jax.Array,
                alpha_optimizer_state: optax.OptState,
            ):
                (
                    new_agent_state,
                    new_optimizer_state,
                    metrics,
                    action_entropy,
                ) = optimizer_step(
                    partial(
                        loss_fn_actor,
                        rng=rng,
                        batch_traj=batch_traj_buffer.data,
                        batch_traj_valid=batch_traj_buffer.data_valid,
                        agent_state=agent_state,
                        log_alpha=log_alpha,
                    ),
                    self.__actor_optimizer,
                    optimizer_state,
                    agent_state,
                    self.__param_mask_actor,
                )

                if self.config.autotune:

                    def alpha_update():
                        def alpha_loss_fn(log_alpha: jax.Array):
                            return jnp.exp(log_alpha) * (
                                action_entropy.mean() - target_entropy
                            )

                        grad_fn = jax.value_and_grad(alpha_loss_fn)
                        alpha_loss, grads = grad_fn(log_alpha)
                        new_log_alpha, new_opt_state_alpha = (
                            self.__alpha_optimizer.update(
                                grads, alpha_optimizer_state, log_alpha
                            )
                        )

                        return (
                            new_log_alpha,
                            new_opt_state_alpha,
                            {
                                "alpha_loss": alpha_loss,
                                "_alpha_loss": True,
                                "target_entropy": target_entropy,
                                "_target_entropy": True,
                            },
                        )

                    def alpha_no_update():
                        return (
                            log_alpha,
                            alpha_optimizer_state,
                            {
                                "alpha_loss": jnp.nan,
                                "_alpha_loss": False,
                                "target_entropy": jnp.nan,
                                "_target_entropy": False,
                            },
                        )

                    if self.config.alpha_update_interval == 1:
                        new_log_alpha, new_opt_state_alpha, alpha_metrics = (
                            alpha_update()
                        )
                    else:
                        actor_update_step = jnp.ceil(
                            gradient_step
                            / self.config.policy_interval
                            * self.config.policy_update_steps
                        ).astype(jnp.int32)
                        new_log_alpha, new_opt_state_alpha, alpha_metrics = (
                            jax.lax.cond(
                                actor_update_step % self.config.alpha_update_interval
                                == 0,
                                alpha_update,
                                alpha_no_update,
                            )
                        )

                    metrics.update(alpha_metrics)
                else:
                    new_opt_state_alpha = alpha_optimizer_state
                    new_log_alpha = log_alpha
                metrics.update({"alpha": jnp.exp(log_alpha), "_alpha": True})

                return (
                    new_agent_state,
                    new_optimizer_state,
                    metrics,
                    new_log_alpha,
                    new_opt_state_alpha,
                )

            def update_target_agent(agent_state: Any, agent_target_state: Any):
                return jax.tree.map(
                    lambda current_state, target_state: self.config.tau * current_state
                    + (1 - self.config.tau) * target_state,
                    agent_state,
                    agent_target_state,
                )

            rng, sub_step_rng = jax.random.split(state.rng)
            gradient_step = training_state.algorithm_state.gradient_step + sub_step
            sub_step_rng, batch_sample_rng = jax.random.split(sub_step_rng)

            batch_traj_buffer = trajectory_buffer.sample_batch(
                batch_sample_rng,
                self.config.batch_size,
                sample_sequence_length,
                context_length=context_length,
            )

            actor_update_rng, critic_update_rng = jax.random.split(sub_step_rng)
            if not self.config.disable_critic_update:
                (
                    new_agent_state,
                    new_opt_state_critic,
                    metrics_critic,
                ) = critic_update(
                    critic_update_rng,
                    state.agent_state,
                    state.critic_optimizer_state,
                )
            else:
                new_agent_state = state.agent_state
                new_opt_state_critic = state.critic_optimizer_state
                metrics_critic = empty_step_metrics["critic"]

            def actor_update_loop_body(_: int, carry):
                (
                    rng,
                    agent_state,
                    actor_optimizer_state,
                    log_alpha,
                    alpha_optimizer_state,
                    _,
                ) = carry
                rng, local_rng = jax.random.split(rng)
                (
                    agent_state,
                    new_opt_state_actor,
                    metrics_actor,
                    new_log_alpha,
                    new_opt_state_alpha,
                ) = actor_update(
                    local_rng,
                    agent_state,
                    actor_optimizer_state,
                    log_alpha,
                    alpha_optimizer_state,
                )
                return (
                    rng,
                    agent_state,
                    new_opt_state_actor,
                    new_log_alpha,
                    new_opt_state_alpha,
                    metrics_actor,
                )

            (
                _,
                new_agent_state,
                new_opt_state_actor,
                new_log_alpha,
                new_opt_state_alpha,
                metrics_actor,
            ) = jax.lax.cond(
                gradient_step % self.config.policy_interval == 0,
                lambda *args: jax.lax.fori_loop(
                    0, self.config.policy_update_steps, actor_update_loop_body, args
                ),
                lambda *args: args,
                actor_update_rng,
                new_agent_state,
                state.actor_optimizer_state,
                state.log_alpha,
                state.alpha_optimizer_state,
                {
                    k: v
                    for k, v in state.metrics["actor"].items()
                    if not k.endswith("loss_divergence")
                    and not re.match(
                        r"_?(actor|prediction)_loss_(train|val)_(mean|std)", k
                    )
                },
            )

            if self.config.use_target_networks:
                new_agent_target_state = jax.lax.cond(
                    gradient_step % self.config.target_network_interval == 0,
                    update_target_agent,
                    lambda agent_state, agent_target_state: agent_target_state,
                    new_agent_state,
                    state.target_agent_state,
                )
            else:
                new_agent_target_state = state.target_agent_state

            output_metrics = {
                "actor": metrics_actor,
                "critic": metrics_critic,
            }

            metrics_step_1 = jax.lax.cond(
                sub_step == 0,
                lambda s: output_metrics,
                lambda s: s.metrics_step_1,
                state,
            )

            return SAC._SubStepState(
                rng,
                new_opt_state_actor,
                new_opt_state_critic,
                new_opt_state_alpha,
                new_agent_state,
                new_agent_target_state,
                output_metrics,
                metrics_step_1,
                new_log_alpha,
            )

        discrete_action_states = jax.tree.reduce(
            lambda a, b: a * b,
            self.__get_discrete_state_counts(self.train_env.single_inner_action_space),
            1,
        )
        continuous_action_dims = jax.tree.reduce(
            lambda a, b: a + b,
            self.__get_continuous_dims(self.train_env.single_inner_action_space),
            0,
        )

        if (
            self.config.discrete_target_entropy_mode
            == DiscreteTargetEntropyMode.DEFAULT
        ):
            target_entropy_scale_discrete = self.__target_entropy_scale_discrete_schedule(
                training_state.algorithm_state.target_entropy_scale_discrete_schedule_state
            )
            discrete_target_entropy = target_entropy_scale_discrete * np.log(
                discrete_action_states
            )
        else:
            epsilon = self.__target_entropy_discrete_epsilon_schedule(
                training_state.algorithm_state.target_entropy_scale_discrete_schedule_state
            )
            discrete_target_entropy = -epsilon * jnp.log(epsilon) - (
                1 - epsilon
            ) * jnp.log((1 - epsilon) / (discrete_action_states - 1))

        target_entropy_scale_continuous = self.__target_entropy_scale_continuous_schedule(
            training_state.algorithm_state.target_entropy_scale_continuous_schedule_state
        )
        continuous_target_entropy = (
            -target_entropy_scale_continuous * continuous_action_dims
        )

        target_entropy = discrete_target_entropy + continuous_target_entropy

        def update(training_state: TrainingState[SACState]):
            training_rng, val_rng = jax.random.split(rng)
            algorithm_state = training_state.algorithm_state

            # Optimizing the policy and value network
            state = SAC._SubStepState(
                training_rng,
                algorithm_state.actor_optimizer_state,
                algorithm_state.critic_optimizer_state,
                algorithm_state.alpha_optimizer_state,
                training_state.agent_state,
                algorithm_state.target_agent_state,
                empty_step_metrics,
                empty_step_metrics,
                algorithm_state.log_alpha,
            )

            current_itr = (
                training_state.total_train_env_steps // self.env_steps_per_iteration
            )

            update_steps = self.__get_total_updates_at_itr(
                current_itr + 1
            ) - self.__get_total_updates_at_itr(current_itr)

            state = jax.lax.fori_loop(0, update_steps, sub_step_fn, state)

            inner_optimizer_state_actor = [
                e
                for e in jax.tree.flatten(
                    state.actor_optimizer_state,
                    is_leaf=lambda n: isinstance(
                        n, optax.InjectStatefulHyperparamsState
                    ),
                )[0]
                if isinstance(e, optax.InjectStatefulHyperparamsState)
            ][0]
            inner_optimizer_state_critic = [
                e
                for e in jax.tree.flatten(
                    state.critic_optimizer_state,
                    is_leaf=lambda n: isinstance(
                        n, optax.InjectStatefulHyperparamsState
                    ),
                )[0]
                if isinstance(e, optax.InjectStatefulHyperparamsState)
            ][0]

            new_training_state = training_state.replace(
                agent_state=state.agent_state,
                algorithm_state=SACState(
                    state.actor_optimizer_state,
                    state.critic_optimizer_state,
                    state.alpha_optimizer_state,
                    state.target_agent_state,
                    state.log_alpha,
                    algorithm_state.gradient_step + update_steps,
                    self.__target_entropy_scale_discrete_schedule.update(
                        algorithm_state.target_entropy_scale_discrete_schedule_state
                    ),
                    self.__target_entropy_discrete_epsilon_schedule.update(
                        algorithm_state.target_entropy_discrete_epsilon_schedule_state
                    ),
                    self.__target_entropy_scale_continuous_schedule.update(
                        algorithm_state.target_entropy_scale_continuous_schedule_state
                    ),
                ),
            )

            # The above as dict
            metrics = {
                "sac": {
                    **state.metrics,
                    "step1": state.metrics_step_1,
                },
                "general": {
                    "learning_rate_actor": inner_optimizer_state_actor.hyperparams[
                        "learning_rate"
                    ],
                    "_learning_rate_actor": True,
                    "learning_rate_critic": inner_optimizer_state_critic.hyperparams[
                        "learning_rate"
                    ],
                    "_learning_rate_critic": True,
                    "update_steps": update_steps,
                    "_update_steps": True,
                    "total_update_steps": new_training_state.algorithm_state.gradient_step,
                    "_total_update_steps": True,
                },
            }

            return new_training_state, metrics

        empty_metrics = self._get_empty_metrics()
        empty_step_metrics = {
            k: v for k, v in empty_metrics["sac"].items() if k not in "step1"
        }

        return update(training_state)

    def __get_total_updates_at_itr(self, itr: jax.Array | int):
        steps = itr * self.env_steps_per_iteration
        if self.config.initial_utd is None:
            update_steps_float = self.config.update_to_data_ratio * steps
        else:
            warmup_steps = (
                self.total_iterations
                * self.config.utd_warmup_steps_rel
                * self.env_steps_per_iteration
            )
            partial_utd_steps = jnp.minimum(warmup_steps, steps)
            full_utd_steps = jnp.maximum(steps - warmup_steps, 0)
            utd_increase_per_itr = (
                self.config.update_to_data_ratio - self.config.initial_utd
            ) / warmup_steps
            update_steps_float = (
                (
                    self.config.initial_utd
                    + 0.5 * utd_increase_per_itr * partial_utd_steps
                )
                * partial_utd_steps
                + full_utd_steps * self.config.update_to_data_ratio
            )
        return jnp.floor(update_steps_float).astype(jnp.int32)

    @staticmethod
    def __get_discrete_state_counts(space: gym.spaces.Space) -> Any:
        return gym_space_map(lambda s: s.n if isinstance(s, Discrete32) else 1, space)

    @staticmethod
    def __get_continuous_dims(space: gym.spaces.Space) -> Any:
        return gym_space_map(
            lambda s: np.prod(s.shape) if isinstance(s, gym.spaces.Box) else 0, space
        )

    @property
    def optimal_sample_sequence_length(self):
        # Try to draw samples such that they are memory aligned to not waste any space in the blocks computed by the
        # attention functions
        memory_alignments = [
            e.memory_alignment
            for e in [
                self.agent.sequence_encoders.actor,
                self.agent.sequence_encoders.critic,
                self.agent.sequence_encoders.predictor,
            ]
        ]
        common_memory_alignment = math.lcm(*memory_alignments)
        # Minimum sample sequence length is self.memory_horizon_with_context + 1 as we need one extra step for value
        # function bootstrapping
        return (
            math.ceil((self.memory_horizon_with_context + 1) / common_memory_alignment)
            * common_memory_alignment
        )
