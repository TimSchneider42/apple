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
from pytree import PyTree
from trajectory_buffer import TrajectoryBuffer, TrajectoryBufferData
from .base_algorithm import (
    BaseAlgorithm,
    AlgorithmSettings,
    EnvironmentTransition,
    TrainingState,
    Trajectory,
    EnvironmentState,
    BaseAlgorithmConfig,
)
from .modules import OptimizerConfig, AgentConfig

ActorMemoryStateType = TypeVar("ActorMemoryStateType")
PredictorMemoryStateType = TypeVar("PredictorMemoryStateType")
CriticMemoryStateType = TypeVar("CriticMemoryStateType")

PPOAgentMemoryStateType = ActorCriticAgentMemoryState[
    ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType
]


class PPOAgent(
    CompositeAgent[ActorCriticAgentStructure],
    BaseAgent[ActorCriticAgentOutput, PPOAgentMemoryStateType],
    Generic[ActorMemoryStateType, PredictorMemoryStateType, CriticMemoryStateType],
):
    heads: ActorCriticAgentStructure[ActorHeadTree, PredictorHeadTree, CriticHead]
    sequence_encoders: ActorCriticAgentStructure[
        MultiModalSequenceEncoder, MultiModalSequenceEncoder, MultiModalSequenceEncoder
    ]


@flax.struct.dataclass
class PPOState:
    optimizer_state: optax.OptState


@flax.struct.dataclass
class PPOTrajectory(Trajectory):
    action_log_prob: PyTree[jax.Array]
    value: jax.Array


@flax.struct.dataclass
class PPOAugmentedTrajectory(PPOTrajectory):
    advantage: jax.Array


# Cannot use generics here for some reason
@dataclass
class PPOConfig(BaseAlgorithmConfig):
    # The discount factor gamma
    gamma: float

    # The lambda for the general advantage estimation
    gae_lambda: float

    # The ratio of update steps to environment steps
    update_to_data_ratio: float

    # Toggles advantages normalization
    norm_adv: bool

    # The surrogate clipping coefficient
    clip_coef: float

    # Toggles whether to use a clipped loss for the value function, as per the paper
    clip_vloss: bool

    # Coefficient of the policy loss
    pg_coef: float

    # Coefficient of the entropy
    ent_coef: float

    # Coefficient of the value function
    vf_coef: float

    # Coefficient of the prediction loss
    p_coef: float

    # The target KL divergence threshold
    target_kl: float | None

    # Number of samples to evaluate in parallel during training
    batch_size: int

    # The number of batched environment steps per iteration
    batch_env_steps_per_iteration: int | None

    # Configuration of the optimizer
    optimizer: OptimizerConfig

    # Configuration of the agent
    agent: AgentConfig

    _type: str = "PPO"  # Do not override

    def make_algorithm(self) -> "PPO":
        return PPO(self)


class PPO(
    BaseAlgorithm[PPOAgent, ActorCriticAgentOutput, PPOState, PPOTrajectory, PPOConfig]
):
    @flax.struct.dataclass
    class _SubStepState:
        rng: jax.Array
        optimizer_state: optax.OptState
        agent_state: Any
        metrics: dict[str, jax.Array]
        metrics_step_1: dict[str, jax.Array]
        clipfracs_sum: jax.Array
        effective_steps: int

    def __init__(self, config: PPOConfig):
        super().__init__("ppo", config=config, on_policy=True)
        self.__optimizer = None

    def __get_initial_algorithm_state(self, initial_agent_state: Any) -> PPOState:
        return PPOState(self.__optimizer.init(initial_agent_state["params"]))

    def _get_initial_algorithm_state(self, initial_agent_state: Any) -> PPOState:
        if (
            self.memory_horizon_with_context
            > self.algorithm_settings.batch_env_steps_per_iteration
        ):
            raise ValueError(
                "The memory horizon with context must be smaller or equal to the batch environment steps collected per"
                "iteration."
            )
        self.__optimizer = self.config.optimizer.make_optimizer(
            self.update_steps * self.total_iterations
        )
        return self.__get_initial_algorithm_state(initial_agent_state)

    def _get_trajectory_variable_spec(self) -> PPOTrajectory:
        return PPOTrajectory.extend(
            super()._get_trajectory_variable_spec(),
            value=jax.ShapeDtypeStruct((), jnp.float32),
            action_log_prob=jax.ShapeDtypeStruct((), jnp.float32),
        )

    def _mk_agent(self, vector_env: BaseActivePerceptionVectorEnv) -> PPOAgent:
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

        return PPOAgent(
            sequence_encoders=sequence_encoders,
            heads=heads,
            observe_actions=self.config.agent.observe_actions,
        )

    def _get_algorithm_settings(self) -> AlgorithmSettings:
        batch_env_steps_per_iteration = self.config.batch_env_steps_per_iteration
        if batch_env_steps_per_iteration is None:
            batch_env_steps_per_iteration = self.max_episode_steps
        return AlgorithmSettings(
            trajectory_buffer_size=batch_env_steps_per_iteration,
            batch_env_steps_per_iteration=batch_env_steps_per_iteration,
        )

    def _get_trajectory_data(
        self, transition: EnvironmentTransition[ActorCriticAgentOutput]
    ) -> dict[str, jax.Array]:
        return PPOTrajectory.extend(
            super()._get_trajectory_data(transition),
            value=transition.agent_output.critic[:, 0],
            action_log_prob=tree_sum(
                transition.action_log_prob, (transition.agent_output.critic.shape[0],)
            ),
        )

    def estimate_advantages(
        self,
        rewards: jax.Array,
        value_estimates: jax.Array,
        terminated: jax.Array,
        truncated: jax.Array,
        next_value_estimate: jax.Array,
        next_terminated: jax.Array,
    ) -> jax.Array:
        CarryType = tuple[jax.Array, jax.Array]

        def scan_func(
            carry: CarryType, args: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[CarryType, jax.Array]:
            lastgaelam, next_value = carry
            reward, value, done = args
            # Since the value of a terminal state is always 0, we do not use the next-state value estimate if it is
            # terminal
            delta = reward + self.config.gamma * next_value - value
            # The advantage of a terminal or truncated state is always 0
            lastgaelam = (
                delta + self.config.gamma * self.config.gae_lambda * lastgaelam
            ) * ~done
            return (lastgaelam, value), lastgaelam

        value_estimates *= ~terminated
        next_value_estimate *= ~next_terminated
        done = terminated | truncated

        _, advantages = jax.lax.scan(
            scan_func,
            (jnp.zeros_like(rewards[:, -1]), next_value_estimate),
            (rewards.T, value_estimates.T, done.T),
            reverse=True,
        )

        return advantages.T

    def _handle_model_reset(
        self, iteration: jax.Array, algorithm_state: PPOState, agent_state: Any
    ) -> PPOState:
        return self.__get_initial_algorithm_state(agent_state)

    def _training_step(
        self,
        iteration: jax.Array,
        rng: jax.Array,
        training_state: TrainingState[PPOState],
        trajectory_buffer: TrajectoryBuffer[PPOTrajectory],
        agent_memory_state: Any,
        current_env_state: EnvironmentState,
    ) -> tuple[TrainingState[PPOState], dict[str, Any]]:
        # Optimizing the policy and value network
        optimizer_state = training_state.algorithm_state.optimizer_state
        agent_state = training_state.agent_state

        # The advantage estimation will fail if the replay buffer is not full
        assert (
            trajectory_buffer.capacity_steps
            == self.algorithm_settings.batch_env_steps_per_iteration
        )
        trajectory_buffer_data = trajectory_buffer.data_aligned
        variables = trajectory_buffer_data.variables

        last_obs = jax.tree.map(lambda x: x[:, -1:], variables.obs)
        last_act = jax.tree.map(lambda x: x[:, -1:], variables.prev_action)
        done = variables.terminated[:, -1:] | variables.truncated[:, -1:]
        rng, eval_rng = jax.random.split(rng)
        agent_output, _ = self.agent.apply(
            training_state.agent_state,
            last_obs,
            last_act,
            done,
            agent_memory_state,
            rngs=eval_rng,
            evaluation_mode=True,
        )

        # TODO: do the advantages have to be updated after every optimization step?
        advantages = self.estimate_advantages(
            variables.reward,
            variables.value,
            variables.terminated,
            variables.truncated,
            agent_output.critic[:, 0],
            current_env_state.terminated,
        )
        augmented_trajectory_buffer_data = TrajectoryBufferData(
            PPOAugmentedTrajectory.extend(trajectory_buffer_data.variables, advantages),
            trajectory_buffer_data.start,
            trajectory_buffer_data.end,
            trajectory_buffer_data.step_no,
            trajectory_buffer_data.trajectory_id,
        )

        augmented_buffer: TrajectoryBuffer[PPOAugmentedTrajectory] = (
            TrajectoryBuffer.from_data(augmented_trajectory_buffer_data)
        )

        augmented_buffer = augmented_buffer.populate_cache(
            self.memory_horizon,
        )

        def sub_step_fn(sub_step: int, state: PPO._SubStepState) -> PPO._SubStepState:
            def sub_step_fn_inner(
                sub_step: int, state: PPO._SubStepState
            ) -> PPO._SubStepState:
                rng, sub_step_rng = jax.random.split(state.rng)
                sub_step_rng, batch_sample_rng = jax.random.split(sub_step_rng)

                context_length = self.memory_horizon_with_context - self.memory_horizon
                batch_traj = augmented_buffer.sample_batch(
                    batch_sample_rng,
                    self.config.batch_size,
                    self.memory_horizon,
                    context_length=context_length,
                )
                v = batch_traj.data

                def loss_fn(params: VariableDict):
                    variables = flax.core.copy(
                        state.agent_state, add_or_replace={"params": params}
                    )
                    (agent_output, _), state_updates = self.agent.apply(
                        variables,
                        v.variables.obs,
                        v.variables.prev_action,
                        v.start,
                        step_no=v.step_no,
                        rngs=sub_step_rng,
                        mutable=["batch_stats"],
                        evaluation_mode=False,
                        norm_mask=batch_traj.data_valid,
                    )
                    value_seq = agent_output.critic
                    log_prob = tree_sum(
                        agent_output.action_distr.log_prob(
                            tree_freeze(v.variables.action)
                        ),
                        value_seq.shape,
                    )
                    log_ratio = log_prob - v.variables.action_log_prob
                    ratio = jnp.exp(log_ratio)

                    advantage = v.variables.advantage
                    batch_returns = v.variables.value * ~v.end + advantage

                    if self.config.norm_adv:
                        advantage = (advantage - advantage.mean()) / (
                            advantage.std() + 1e-8
                        )

                    done = v.variables.terminated | v.variables.truncated

                    # Policy loss
                    pg_loss1 = -advantage * ratio
                    pg_loss2 = -advantage * jnp.clip(
                        ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef
                    )
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean(where=~done)
                    # Value loss
                    if self.config.clip_vloss:
                        v_loss_unclipped = (value_seq - batch_returns) ** 2
                        v_clipped = v.variables.value + jnp.clip(
                            value_seq - v.variables.value,
                            -self.config.clip_coef,
                            self.config.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - batch_returns) ** 2
                        v_loss_arr = jnp.maximum(v_loss_unclipped, v_loss_clipped)
                    else:
                        v_loss_arr = (value_seq - batch_returns) ** 2
                    v_loss = 0.5 * v_loss_arr.mean(where=~done)

                    prediction_loss = self.train_env.loss_fn.jax(
                        agent_output.prediction,
                        v.variables.next_metadata.perception_target,
                        value_seq.shape,
                    ).mean()

                    entropy_loss = -tree_sum(
                        agent_output.action_distr.entropy(), value_seq.shape
                    ).mean()
                    metrics = {
                        "entropy_loss": entropy_loss,
                        "pg_loss": pg_loss,
                        "v_loss": v_loss,
                        "p_loss": prediction_loss,
                        "log_ratio": log_ratio,
                        "ratio": ratio,
                    }
                    loss = (
                        self.config.pg_coef * pg_loss
                        + self.config.ent_coef * entropy_loss
                        + self.config.vf_coef * v_loss
                        + self.config.p_coef * prediction_loss
                    )
                    return loss, (metrics, state_updates)

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, (metrics, state_updates)), grads = grad_fn(
                    state.agent_state["params"]
                )
                new_params, new_opt_state = self.__optimizer.update(
                    grads, state.optimizer_state, state.agent_state["params"]
                )
                new_agent_state = flax.core.copy(
                    state.agent_state,
                    add_or_replace={"params": new_params, **state_updates},
                )

                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = -metrics["log_ratio"].mean()
                approx_kl = ((metrics["ratio"] - 1) - metrics["log_ratio"]).mean()
                clipfracs_sum = (
                    state.clipfracs_sum
                    + (jnp.abs(metrics["ratio"] - 1.0) > self.config.clip_coef)
                    .astype(jnp.float32)
                    .mean()
                )

                output_metrics = {
                    "entropy_loss": metrics["entropy_loss"],
                    "policy_loss": metrics["pg_loss"],
                    "value_loss": metrics["v_loss"],
                    "prediction_loss": metrics["p_loss"],
                    "old_approx_kl": approx_kl,
                    "approx_kl": old_approx_kl,
                }
                metrics_step_1 = jax.lax.cond(
                    sub_step == 0,
                    lambda s: output_metrics,
                    lambda s: s.metrics_step_1,
                    state,
                )

                return PPO._SubStepState(
                    rng,
                    new_opt_state,
                    new_agent_state,
                    output_metrics,
                    metrics_step_1,
                    clipfracs_sum,
                    sub_step,
                )

            break_cond = (
                self.config.target_kl is not None
                and state.metrics.get("approx_kl", 0.0) > self.config.target_kl
            )
            return jax.lax.cond(
                break_cond,
                lambda sub_step, state: state,
                sub_step_fn_inner,
                sub_step,
                state,
            )

        metrics_init = {
            "entropy_loss": jnp.nan,
            "policy_loss": jnp.nan,
            "value_loss": jnp.nan,
            "old_approx_kl": jnp.nan,
            "prediction_loss": jnp.nan,
            "approx_kl": jnp.nan,
        }

        state = PPO._SubStepState(
            rng, optimizer_state, agent_state, metrics_init, metrics_init, 0.0, 0
        )
        state = jax.lax.fori_loop(0, self.update_steps, sub_step_fn, state)

        y_pred = augmented_trajectory_buffer_data.variables.value
        y_true = (
            augmented_trajectory_buffer_data.variables.advantage
            + augmented_trajectory_buffer_data.variables.value
        )
        var_y = jnp.var(y_true)
        explained_var = jax.lax.select(
            var_y == 0, jnp.nan, 1 - jnp.var(y_true - y_pred) / var_y
        )

        inner_optimizer_state = jax.tree.flatten(
            optimizer_state,
            is_leaf=lambda n: isinstance(n, optax.InjectStatefulHyperparamsState),
        )[0][0]

        # The above as dict
        metrics = {
            "ppo": {
                **state.metrics,
                "step1": state.metrics_step_1,
                "clipfrac": state.clipfracs_sum / state.effective_steps,
                "explained_variance": explained_var,
            },
            "general": {
                "learning_rate": inner_optimizer_state.hyperparams["learning_rate"],
            },
        }

        training_state = training_state.replace(
            agent_state=state.agent_state,
            algorithm_state=PPOState(state.optimizer_state),
        )
        return training_state, metrics

    @property
    def update_steps(self):
        return int(
            round(self.config.update_to_data_ratio * self.env_steps_per_iteration)
        )
