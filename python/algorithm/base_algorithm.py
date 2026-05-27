from __future__ import annotations

import copy
import logging
import math
import re
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Generic, TypeVar, Any, Literal, Mapping, Iterable

import flax.struct
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax.core.scope import VariableDict
from flax.cursor import cursor
from gymnasium.vector import VectorEnv
from orbax.checkpoint.utils import from_flat_dict, to_flat_dict

from agent import BaseAgentOutput, BaseAgent
from ap_gym import (
    BaseActivePerceptionVectorEnv,
    CrossEntropyLossFn,
    ensure_active_perception_vector_env,
)
from data_logger import DataLogger, BaseDataLogger
from gym_space_map import gym_space_map
from hostcallback import io_callback, DataTransferMode
from jax_rate import JaxRate
from jax_util import tree_unfreeze, tree_freeze, compute_param_metrics
from metric_log_level import MetricLogLevel
from pytree import PyTree
from ring_buffer import RingBuffer
from trajectory_buffer import TrajectoryBuffer, TrajectoryBufferData
from util import fix_ordered_dicts, closest_prime, get_max_episode_steps
from .util import (
    add_seq_dim,
    generate_log_dict,
    merge_dicts,
    rem_seq_dim,
    mk_seed,
    trajectory_variable_spec_from_space,
)
from .wrappers import (
    CheckMaxStepsVectorWrapper,
    LogWrapper,
    JaxWrapper,
    GymVectorWrapper32,
    ActivePerceptionMetadata,
    MaskActivePerceptionEnvVectorWrapper,
    AddRenderObservationVectorWrapper,
    ClassificationBinaryRewardVectorWrapper,
    RandomActionWrapper,
    Grid2DActionWrapper,
    DetectImageObsWrapper,
)

logger = logging.getLogger(__name__)

AgentOutputType = TypeVar("AgentOutputType", bound=BaseAgentOutput)
AgentType = TypeVar("AgentType", bound=BaseAgent)
AlgorithmStateType = TypeVar("AlgorithmStateType")
TrajectoryStepType = TypeVar("TrajectoryStepType", bound="Trajectory")


@dataclass(frozen=True)
class AlgorithmSettings:
    trajectory_buffer_size: int
    batch_env_steps_per_iteration: int
    # Minimum number of steps before learning starts, independently of learning_starts setting
    min_learning_starts: int = 0


@flax.struct.dataclass
class EnvironmentState:
    obs: PyTree[jax.Array]
    metadata: ActivePerceptionMetadata | None
    terminated: jax.Array
    truncated: jax.Array
    episode_step_count: jax.Array


@flax.struct.dataclass
class EnvironmentTransition(Generic[AgentOutputType]):
    prev_action: PyTree[jax.Array]
    prev_state: EnvironmentState
    agent_output: AgentOutputType
    action: PyTree[jax.Array]
    action_log_prob: PyTree[jax.Array]
    reward: jax.Array
    next_state: EnvironmentState


@flax.struct.dataclass
class Trajectory:
    prev_action: PyTree[jax.Array]
    obs: PyTree[jax.Array]
    terminated: jax.Array
    truncated: jax.Array
    action: PyTree[jax.Array]
    reward: jax.Array
    next_metadata: ActivePerceptionMetadata

    @classmethod
    def extend(cls, base: "Trajectory", *args, **kwargs):
        return cls(
            *(base.__dict__[f] for f in base.__dataclass_fields__), *args, **kwargs
        )


@flax.struct.dataclass
class TrainingState(Generic[AlgorithmStateType]):
    algorithm_state: AlgorithmStateType
    agent_state: Any
    total_train_env_steps: int


@flax.struct.dataclass
class _PolicyInput:
    obs: PyTree[jax.Array]
    prev_action: PyTree[jax.Array]
    episode_first_step: jax.Array


@flax.struct.dataclass
class _StepState:
    rng: jax.Array
    env_state: EnvironmentState
    episode_first_step: jax.Array
    action: PyTree[jax.Array]
    agent_memory_state: Any
    total_environment_steps: int
    trajectory_buffer: TrajectoryBuffer | None = None


@flax.struct.dataclass
class _IterationState:
    rng: jax.Array
    prev_step_state: _StepState
    agent_state: Any
    log_rate: JaxRate


@flax.struct.dataclass
class _IterationStateTrain(_IterationState, Generic[AlgorithmStateType]):
    algorithm_state: AlgorithmStateType


_IterationStateType = TypeVar("_IterationStateType", bound=_IterationState)


@flax.struct.dataclass
class _MainState:
    iteration_state_train: _IterationStateTrain
    iteration_state_eval: _IterationState | None = None


@flax.struct.dataclass
class TrainingOutput:
    agent_state: VariableDict
    replay_buffer: TrajectoryBuffer | None = None


class ReplayBufferImageDType(str, Enum):
    F32 = "f32"
    F16 = "f16"
    U8 = "u8"


class ParameterType(str, Enum):
    UNNORMALIZED_WEIGHTS = "unnormalized_weights"
    NORMALIZED_WEIGHTS = "normalized_weights"
    UNNORMALIZED_SCALES_AND_BIASES = "unnormalized_scales_and_biases"
    NORMALIZED_SCALES_AND_BIASES = "normalized_scales_and_biases"
    OTHER = "other"


class ControlStrategy(str, Enum):
    POLICY = "policy"
    RANDOM = "random"
    GRID_2D = "grid_2d"


AlgorithmType = TypeVar("AlgorithmType", bound="BaseAlgorithm")


@dataclass
class BaseAlgorithmConfig(ABC, Generic[AlgorithmType]):
    # Maximum number of steps the environment is allowed to take before it must reset
    max_episode_steps: int | None

    # Total number of timesteps for training
    total_env_steps: int

    # Number of steps to take in the environment in this training segment
    env_steps: int | None

    # Evaluate the agent every n episodes on the evaluation environment if given
    eval_every: int

    # Minimum standard deviation of the observation normalizer
    obs_norm_min_std: float

    # Minimum standard deviation of the action normalizer
    act_norm_min_std: float

    # If set, the environment will be reset between episodes
    reset_env_between_iterations: bool

    # Number of videos to log during training
    video_log_count: int

    # Number of steps for which statistics get logged during training
    stat_log_count: int

    # Number of steps for which metrics get logged during training
    metrics_log_count: int

    # Smoothing factor used for printing the rewards
    print_smoothing_factor: float

    # Interval in which to print the rewards
    print_interval: int

    # Which control strategy to use. Options are
    # - POLICY: use the policy of the agent
    # - RANDOM: use random actions
    # - GRID_2D: use a 2D grid motion pattern
    control_strategy: ControlStrategy

    # Maximum side length of the images stored in the replay buffer
    replay_buffer_max_image_size: int | None

    # Datatype of the images stored in the replay buffer
    replay_buffer_img_dtype: ReplayBufferImageDType

    # Number of steps before the agent starts learning
    learning_starts: int

    # Number of intermediate checkpoints to keep
    intermediate_checkpoint_count: int

    # If set, the agent will save checkpoints during training
    save_checkpoints: bool

    # If set, the agent will store the replay buffer in the checkpoint
    checkpoint_store_trajectory_buffer: bool

    # Number of intermediate model resets to perform during training
    num_intermediate_model_resets: int

    # Maximum memory horizon to use in the encoder model. Is ignored by LSTM model (which might make it act weird, as
    # it will be rolled out on longer sequences than it was trained on). If None, the full episode is used.
    max_memory_horizon: int | None

    # Whether to profile JAX
    profile: bool

    # Whether to create a Perfetto link when profiling
    profile_create_perfetto_link: bool

    # Postprocess the classification reward to be 0/1 instead of the cross-entropy
    classification_binary_reward: bool

    # If set, the environments will be treated as if they were regular gymnasium environments. Effectively, for ap_gym
    # environments, it will treat the prediction output as extra action components, and ignore the loss function. To
    # facilitate critic hints, the perception target space will be retained.
    treat_ap_gym_env_as_gym_env: bool

    # Whether to use the rendering of the environment as observation
    use_rendering_as_observation: bool

    # To which degree to log metrics
    metric_log_level: MetricLogLevel

    # Whether to interpret the final (timeout) truncation in an episode as a termination signal for the agent.
    interpret_final_truncation_as_termination: bool

    @abstractmethod
    def make_algorithm(self) -> AlgorithmType:
        pass


ConfigType = TypeVar("ConfigType", bound=BaseAlgorithmConfig)


class ContextVar:
    pass


class ContextNotInitializedError(Exception):
    pass


class ContextVariableReadOnlyError(AttributeError):
    pass


class BaseAlgorithm(
    ABC,
    Generic[
        AgentType, AgentOutputType, AlgorithmStateType, TrajectoryStepType, ConfigType
    ],
):
    agent: AgentType = ContextVar()
    train_env: JaxWrapper = ContextVar()
    eval_env: JaxWrapper | None = ContextVar()
    max_episode_steps: int = ContextVar()
    algorithm_settings: AlgorithmSettings = ContextVar()
    __checkpoint_manager: ocp.CheckpointManager = ContextVar()
    __data_logger: DataLogger = ContextVar()
    __train_logger: BaseDataLogger = ContextVar()
    __eval_logger: BaseDataLogger = ContextVar()

    __context: Literal["initializing", "initialized"] = None
    __context_variables: tuple[str, ...] = None

    def __init__(self, name: str, config: ConfigType, on_policy: bool = False):
        self.__name = name
        self.__config = config
        self.__on_policy = on_policy

    @abstractmethod
    def _get_initial_algorithm_state(
        self, initial_agent_state: Any
    ) -> AlgorithmStateType:
        pass

    @abstractmethod
    def _get_algorithm_settings(self) -> AlgorithmSettings:
        pass

    @abstractmethod
    def _mk_agent(self, vector_env: BaseActivePerceptionVectorEnv) -> AgentType:
        pass

    def _get_trajectory_variable_spec(self) -> TrajectoryStepType:
        return Trajectory(
            prev_action=trajectory_variable_spec_from_space(
                self.train_env.single_inner_action_space
            ),
            obs=trajectory_variable_spec_from_space(
                self.train_env.single_observation_space
            ),
            terminated=jax.ShapeDtypeStruct((), jnp.bool_),
            truncated=jax.ShapeDtypeStruct((), jnp.bool_),
            action=trajectory_variable_spec_from_space(
                self.train_env.single_inner_action_space
            ),
            reward=jax.ShapeDtypeStruct((), jnp.float32),
            next_metadata=self.train_env.perception_metadata_shape_single,
        )

    def _training_step(
        self,
        iteration: jax.Array,
        rng: jax.Array,
        training_state: TrainingState[AlgorithmStateType],
        trajectory_buffer: TrajectoryBuffer[TrajectoryStepType],
        agent_memory_state: Any,
        current_env_state: EnvironmentState,
    ) -> tuple[TrainingState[AlgorithmStateType], dict[str, Any]]:
        pass

    def _get_trajectory_data(
        self, transition: EnvironmentTransition[AgentOutputType]
    ) -> TrajectoryStepType:
        return Trajectory(
            prev_action=transition.prev_action,
            obs=transition.prev_state.obs,
            terminated=transition.prev_state.terminated,
            truncated=transition.prev_state.truncated,
            action=transition.action,
            reward=transition.reward,
            next_metadata=transition.next_state.metadata,
        )

    def _get_empty_metrics(self) -> dict[str, Any] | None:
        return None

    def _process_initial_agent_state(
        self, initial_agent_state: dict[str, Any]
    ) -> dict[str, Any]:
        return initial_agent_state

    def _reset_env(
        self,
        env: JaxWrapper,
        rng: jax.Array,
        agent_state: Any,
        trajectory_buffer: TrajectoryBuffer[TrajectoryStepType] | None = None,
        set_env_seed: bool = False,
    ) -> _StepState:
        step_rng, init_memory_rng, reset_rng = jax.random.split(rng, num=3)
        seed = mk_seed(reset_rng) if set_env_seed else None
        ret = env.reset_jax(seed)
        terminated = truncated = jnp.zeros(env.num_envs, dtype=jnp.bool_)

        single_inner_action_space_pytree = gym_space_map(
            lambda x: x, env.single_action_space["action"]
        )
        action = jax.tree.map(
            lambda s: jnp.zeros((env.num_envs, *s.shape), s.dtype),
            single_inner_action_space_pytree,
        )
        agent_memory_state = self.agent.apply(
            agent_state,
            (env.num_envs,),
            method=self.agent.init_memory,
            rngs=init_memory_rng,
        )

        state = EnvironmentState(
            ret.obs,
            ActivePerceptionMetadata(
                jnp.zeros(env.num_envs),
                jax.tree.map(
                    lambda x: jnp.zeros_like(x), env.prediction_target_space.sample()
                ),
            ),
            terminated,
            truncated,
            jnp.zeros(env.num_envs, dtype=jnp.int32),
        )

        # Mark the previous episodes as truncated
        if trajectory_buffer is not None:
            old_truncated = jnp.ones(env.num_envs, dtype=jnp.bool_)
            new_variables = jax.tree.map(
                lambda x: None, trajectory_buffer[-1].variables
            ).replace(truncated=old_truncated)
            # Does not cause problems if length == 0, so we avoid a conditional here
            trajectory_buffer = trajectory_buffer.replace_last_step(
                old_truncated, new_variables
            )

        return _StepState(
            step_rng,
            state,
            jnp.ones(env.num_envs, dtype=jnp.bool_),
            action,
            agent_memory_state,
            ret.total_environment_steps,
            trajectory_buffer=trajectory_buffer,
        )

    def _step_env(
        self,
        env: JaxWrapper,
        agent_state: Any,
        step_state: _StepState,
        evaluation_mode: bool,
    ) -> _StepState:
        new_rng, action_rng, eval_rng = jax.random.split(step_state.rng, 3)

        done = step_state.env_state.terminated | step_state.env_state.truncated

        agent_output, new_agent_memory_state = self.agent.apply(
            agent_state,
            add_seq_dim(step_state.env_state.obs),
            add_seq_dim(step_state.action),
            add_seq_dim(step_state.episode_first_step),
            step_state.agent_memory_state,
            rngs=eval_rng,
            evaluation_mode=True,
        )

        def get_random_action(seed: jax.Array) -> PyTree[jax.Array]:
            full_action = env.sample_random_action(rng=seed)
            action = tree_freeze(
                jax.tree.map(lambda x: x[:, None], full_action["action"])
            )
            return action, agent_output.action_distr.log_prob(action)

        new_action, action_log_prob = jax.lax.cond(
            evaluation_mode
            or step_state.total_environment_steps >= self.learning_starts,
            lambda seed: agent_output.action_distr.sample_and_log_prob(seed=seed),
            get_random_action,
            action_rng,
        )
        new_action = rem_seq_dim(tree_unfreeze(new_action))
        action_log_prob = rem_seq_dim(action_log_prob)

        ret = env.step_jax(
            new_action, rem_seq_dim(tree_unfreeze(agent_output.prediction))
        )

        new_step_count = (step_state.env_state.episode_step_count + 1) * ~done
        if self.config.interpret_final_truncation_as_termination:
            converted_truncation = (
                ret.truncated & new_step_count == self.max_episode_steps
            )
            truncated = ret.truncated & ~converted_truncation
            terminated = ret.terminated | converted_truncation
        else:
            terminated = ret.terminated
            truncated = ret.truncated

        new_state = EnvironmentState(
            ret.obs,
            ret.metadata,
            terminated,
            truncated,
            new_step_count,
        )

        if not evaluation_mode:
            transition = EnvironmentTransition(
                step_state.action,
                step_state.env_state,
                agent_output,
                new_action,
                action_log_prob,
                ret.reward,
                new_state,
            )
            trajectory_data = self._get_trajectory_data(transition)
            new_trajectory_buffer = step_state.trajectory_buffer.add_step(
                done=done, variables=trajectory_data
            )
        else:
            new_trajectory_buffer = step_state.trajectory_buffer

        next_step_is_first = done

        output = _StepState(
            new_rng,
            new_state,
            next_step_is_first,
            new_action,
            new_agent_memory_state,
            ret.total_environment_steps,
            trajectory_buffer=new_trajectory_buffer,
        )

        return output

    def iteration_fn(
        self,
        iteration: jax.Array,
        env: JaxWrapper,
        evaluation_mode: bool,
        data_logger: BaseDataLogger,
        iteration_state: _IterationStateType,
    ) -> _IterationStateType:
        env_type = "eval" if evaluation_mode else "train"
        new_rng, iteration_rng = jax.random.split(iteration_state.rng)

        agent_state = iteration_state.agent_state
        step_rng, iteration_rng = jax.random.split(iteration_rng)
        step_state: _StepState = iteration_state.prev_step_state.replace(rng=step_rng)
        if self.__on_policy and step_state.trajectory_buffer is not None:
            step_state = step_state.replace(
                trajectory_buffer=step_state.trajectory_buffer.clear()
            )

        if self.config.reset_env_between_iterations:
            step_state = self._reset_env(
                env,
                step_state.rng,
                agent_state,
                step_state.trajectory_buffer,
            )

        step_state = jax.lax.fori_loop(
            0,
            self.algorithm_settings.batch_env_steps_per_iteration,
            lambda _, s: self._step_env(
                env=env,
                agent_state=agent_state,
                evaluation_mode=evaluation_mode,
                step_state=s,
            ),
            step_state,
        )

        metrics = {f"{env_type}_insights": {}, env_type: {}}

        if isinstance(iteration_state, _IterationStateTrain):
            training_state = TrainingState(
                iteration_state.algorithm_state,
                agent_state,
                step_state.total_environment_steps,
            )
            args = (
                iteration,
                iteration_rng,
                training_state,
                step_state.trajectory_buffer,
                iteration_state.prev_step_state.agent_memory_state,
                step_state.env_state,
            )
            if not evaluation_mode:
                if self.learning_starts > 0:
                    empty_metrics = self._get_empty_metrics()
                    if empty_metrics is None:
                        raise ValueError(
                            "Empty metrics must be provided if learning_starts > 0"
                        )
                    new_training_state, algorithm_metrics = jax.lax.cond(
                        step_state.total_environment_steps >= self.learning_starts,
                        self._training_step,
                        lambda *args, **kwargs: (training_state, empty_metrics),
                        *args,
                    )
                else:
                    new_training_state, algorithm_metrics = self._training_step(*args)
            else:
                new_training_state = training_state
                algorithm_metrics = self._get_empty_metrics()

            agent_state = new_training_state.agent_state
            algorithm_state = new_training_state.algorithm_state
            metrics = merge_dicts(metrics, algorithm_metrics)
        else:
            algorithm_state = None

        @io_callback(
            data_transfer_mode_device_to_host=DataTransferMode.PACKED, ordered=True
        )
        def log_metrics(
            metrics: dict[str, Any],
            total_env_steps: int,
        ):
            data_logger.write(generate_log_dict(metrics), total_env_steps.item())

        log_rate, _ = iteration_state.log_rate(
            step_state.total_environment_steps,
            log_metrics,
            lambda metrics, total_env_steps: None,
            metrics=metrics,
            total_env_steps=step_state.total_environment_steps,
        )

        if isinstance(iteration_state, _IterationStateTrain):
            return _IterationStateTrain(
                rng=new_rng,
                prev_step_state=step_state,
                agent_state=agent_state,
                log_rate=log_rate,
                algorithm_state=algorithm_state,
            )
        else:
            return _IterationState(
                rng=new_rng,
                prev_step_state=step_state,
                agent_state=agent_state,
                log_rate=log_rate,
            )

    def _reset_envs_post_checkpoint(
        self,
        state: _MainState,
    ) -> _MainState:
        # We need to reset all environments, as otherwise we will not have a deterministic continuation
        itr_state_train = state.iteration_state_train
        new_step_state_train = self._reset_env(
            self.train_env,
            itr_state_train.rng,
            itr_state_train.agent_state,
            trajectory_buffer=itr_state_train.prev_step_state.trajectory_buffer,
            set_env_seed=True,
        )
        new_itr_state_train = itr_state_train.replace(
            prev_step_state=new_step_state_train,
        )

        itr_state_eval = state.iteration_state_eval
        if itr_state_eval is not None:
            new_step_state_eval = self._reset_env(
                self.eval_env,
                itr_state_eval.rng,
                itr_state_eval.agent_state,
                trajectory_buffer=itr_state_eval.prev_step_state.trajectory_buffer,
                set_env_seed=True,
            )
            new_itr_state_eval = itr_state_eval.replace(
                prev_step_state=new_step_state_eval,
            )
        else:
            new_itr_state_eval = None
        return _MainState(new_itr_state_train, new_itr_state_eval)

    def _load_checkpoint(
        self,
        iteration: int | jax.Array,
        state: _MainState,
        retain_data_logger_step: bool = False,
    ):
        @io_callback(result_shape=state)
        def load_checkpoint_cb(iteration: int, state: _MainState):
            print(f"Loading checkpoint of iteration {iteration}... ")
            checkpoint_dict = self.__checkpoint_manager.restore(
                iteration,
                args=ocp.args.StandardRestore(
                    fallback_sharding=jax.sharding.SingleDeviceSharding(
                        jax.devices("cpu")[0]
                    )
                ),
            )
            prev_step_state_train = checkpoint_dict["state"]["iteration_state_train"][
                "prev_step_state"
            ]
            prev_step_state_eval = checkpoint_dict["state"]["iteration_state_eval"][
                "prev_step_state"
            ]
            trajectory_buffer_dict = prev_step_state_train["trajectory_buffer"]
            num_streams, capacity = trajectory_buffer_dict["data"]["start"].shape
            if num_streams != self.train_env.num_envs:
                if num_streams < self.train_env.num_envs:
                    raise ValueError(
                        "Cannot increase number of environments when resuming from a checkpoint."
                    )
                else:

                    logger.warning(
                        f"Reducing trajectory buffer streams from {num_streams} to {self.train_env.num_envs}."
                    )

                    def remove_streams(x: jax.Array) -> jax.Array:
                        if x.shape[0:1] == (num_streams,):
                            output = x[: self.train_env.num_envs]
                            x.delete()
                            return output
                        return x

                    trajectory_buffer_dict["trajectory_info"] = jax.tree.map(
                        remove_streams, trajectory_buffer_dict["trajectory_info"]
                    )
                    trajectory_buffer_dict["data"] = jax.tree.map(
                        remove_streams, trajectory_buffer_dict["data"]
                    )
                    for prev_step_state in [
                        prev_step_state_train,
                        prev_step_state_eval,
                    ]:
                        prev_step_state["episode_first_step"] = remove_streams(
                            prev_step_state["episode_first_step"]
                        )
                        prev_step_state["env_state"] = jax.tree.map(
                            remove_streams, prev_step_state["env_state"]
                        )
                        prev_step_state["action"] = jax.tree.map(
                            remove_streams, prev_step_state["action"]
                        )
                        prev_step_state["agent_memory_state"] = jax.tree.map(
                            remove_streams, prev_step_state["agent_memory_state"]
                        )

            if capacity != self.algorithm_settings.trajectory_buffer_size:

                def resize_data(x: jax.Array) -> jax.Array:
                    part1 = x[
                        :,
                        trajectory_buffer_dict["start_index"] : trajectory_buffer_dict[
                            "start_index"
                        ]
                        + new_length,
                    ]
                    part2 = x[:, : (new_length - part1.shape[1])]
                    x.delete()
                    return jnp.concatenate([part1, part2], axis=1)

                new_length = jnp.minimum(
                    trajectory_buffer_dict["length"],
                    self.algorithm_settings.trajectory_buffer_size,
                )
                new_start_index = jnp.zeros_like(trajectory_buffer_dict["start_index"])
                new_data = jax.tree.map(resize_data, trajectory_buffer_dict["data"])

                trajectory_buffer_dict.update(
                    TrajectoryBuffer(
                        TrajectoryBufferData(**new_data),
                        new_start_index,
                        new_length,
                        None,
                    )
                    .rebuild_trajectory_info_buffer()
                    .__dict__
                )

            checkpoint = from_flat_dict(
                to_flat_dict(checkpoint_dict),
                target={"state": state, "data_logger_step": 0},
            )

            checkpoint_state = checkpoint["state"]
            self.train_env.env.total_environment_steps = (
                checkpoint_state.iteration_state_train.prev_step_state.total_environment_steps.item()
            )
            if (
                self.eval_env is not None
                and checkpoint_state.iteration_state_eval is not None
            ):
                self.eval_env.env.total_environment_steps = (
                    checkpoint_state.iteration_state_eval.prev_step_state.total_environment_steps.item()
                )
            if not retain_data_logger_step:
                # Advance the data logger to the checkpoint step
                self.__data_logger.write({}, checkpoint["data_logger_step"])
            print("Done loading checkpoint.")
            return checkpoint_state

        return self._reset_envs_post_checkpoint(load_checkpoint_cb(iteration, state))

    def _save_checkpoint(
        self,
        iteration: jax.Array,
        state: _MainState,
    ) -> _MainState:
        @io_callback(ordered=True)
        def save_checkpoint_cb(iteration: jax.Array, state: _MainState):
            iteration = iteration.item()
            if not self.config.checkpoint_store_trajectory_buffer:
                state = cursor(
                    state
                ).iteration_state_train.prev_step_state.trajectory_buffer.set(None)
            print(f"Saving checkpoint of iteration {iteration}... ")
            self.__checkpoint_manager.save(
                iteration,
                {"state": state, "data_logger_step": self.__data_logger.current_step},
            )
            print("Done saving checkpoint.")

        save_checkpoint_cb(iteration, state)
        return self._reset_envs_post_checkpoint(state)

    def __get_all_instance_variables(self):
        all_statics = {}
        for cls in self.__class__.__mro__:
            for k, v in cls.__dict__.items():
                if not callable(v):
                    all_statics[k] = v

        all_dynamics = self.__dict__

        all_members = {**all_statics, **all_dynamics}
        return tuple(k for k, v in all_members.items() if isinstance(v, ContextVar))

    @contextmanager
    def __setup(
        self,
        env: VectorEnv,
        data_logger: DataLogger,
        eval_env: VectorEnv | None = None,
    ):
        assert self.__context is None
        c_self: BaseAlgorithm[
            AgentType,
            AgentOutputType,
            AlgorithmStateType,
            TrajectoryStepType,
            ConfigType,
        ] = copy.deepcopy(self)
        c_self.__context = "initializing"
        c_self.__context_variables = c_self.__get_all_instance_variables()
        c_self.__data_logger = data_logger

        with c_self.__data_logger:
            c_self.__train_logger = c_self.__data_logger.custom_axis("train_env_steps")
            train_smoothing_factor = c_self.config.print_smoothing_factor
            train_print_interval = c_self.config.print_interval
            if c_self.config.stat_log_count > 0:
                train_stats_log_interval = closest_prime(
                    c_self.config.total_env_steps // c_self.config.stat_log_count
                )
            else:
                train_stats_log_interval = None
            if c_self.config.video_log_count > 0:
                train_video_log_interval = (
                    c_self.config.total_env_steps // c_self.config.video_log_count
                )
            else:
                train_video_log_interval = None
            c_self.max_episode_steps = c_self.config.max_episode_steps
            if c_self.max_episode_steps is None:
                c_self.max_episode_steps = get_max_episode_steps(env)
            if c_self.max_episode_steps is None:
                raise ValueError(
                    "Could not read max_episode_steps from environment or config. Please specify it "
                    "by setting max_episode_steps."
                )
            c_self.train_env = c_self.__prepare_env(
                env,
                log_wrapper_kwargs=dict(
                    data_logger=c_self.__train_logger,
                    prefix="train_env",
                    print_interval=train_print_interval,
                    smoothing_factor=train_smoothing_factor,
                    video_log_interval=train_video_log_interval,
                    stat_log_interval=train_stats_log_interval,
                    return_log_interval=train_stats_log_interval,
                ),
            )
            if eval_env is not None:
                c_self.__eval_logger = c_self.__data_logger.custom_axis(
                    "eval_env_steps", display_axis_name="train_env_steps"
                )
                eval_print_interval = max(
                    int(train_print_interval / c_self.config.eval_every), 1
                )
                if train_video_log_interval is None:
                    eval_video_log_interval = None
                else:
                    eval_video_log_interval = max(
                        int(train_video_log_interval / c_self.config.eval_every), 1
                    )
                eval_smoothing_factor = train_smoothing_factor**c_self.config.eval_every
                c_self.eval_env = c_self.__prepare_env(
                    eval_env,
                    log_wrapper_kwargs=dict(
                        data_logger=c_self.__eval_logger,
                        prefix="eval_env",
                        print_interval=eval_print_interval,
                        smoothing_factor=eval_smoothing_factor,
                        video_log_interval=eval_video_log_interval,
                        stat_log_interval=0,
                        return_log_interval=0,
                    ),
                )
            else:
                c_self.eval_env = None

            c_self.agent = c_self._mk_agent(c_self.train_env)
            c_self.algorithm_settings = c_self._get_algorithm_settings()

            checkpoint_dir = (
                c_self.__data_logger.run_directory / "checkpoints"
            ).resolve()
            options = ocp.CheckpointManagerOptions(
                create=True,
            )
            async_checkpointer = ocp.AsyncCheckpointer(ocp.StandardCheckpointHandler())
            with ocp.CheckpointManager(
                checkpoint_dir,
                async_checkpointer,
                options,
            ) as c_self.__checkpoint_manager:
                c_self.__context = "initialized"
                yield c_self

    def _mk_initial_state(
        self, rng: jax.Array, create_trajectory_buffer: bool = True
    ) -> _MainState:
        new_rng, rng_agent_init, train_rng, reset_rng_train = jax.random.split(rng, 4)

        if create_trajectory_buffer:
            trajectory_buffer = TrajectoryBuffer.build(
                self._get_trajectory_variable_spec(),
                capacity=self.algorithm_settings.trajectory_buffer_size,
                stream_count=self.train_env.num_envs,
            )
        else:
            trajectory_buffer = None

        assert isinstance(self.train_env.action_space, gym.spaces.Dict)
        act_sample = fix_ordered_dicts(self.train_env.inner_action_space.sample())
        obs_sample = fix_ordered_dicts(self.train_env.observation_space.sample())
        initial_agent_state = self.agent.init(
            rng_agent_init,
            add_seq_dim(obs_sample),
            add_seq_dim(act_sample),
            jnp.ones((self.train_env.num_envs, 1), dtype=jnp.bool_),
            evaluation_mode=True,
        )

        if self.config.metrics_log_count > 0:
            log_rate = JaxRate.build(
                closest_prime(
                    self.config.total_env_steps // self.config.metrics_log_count
                )
            )
        else:
            log_rate = JaxRate.never()

        iteration_state_train = _IterationStateTrain(
            rng=train_rng,
            prev_step_state=self._reset_env(
                self.train_env,
                reset_rng_train,
                initial_agent_state,
                trajectory_buffer,
                set_env_seed=True,
            ),
            agent_state=initial_agent_state,
            algorithm_state=self._get_initial_algorithm_state(initial_agent_state),
            # Using the closest prime here as different metrics get logged in different intervals, and we do not want to
            # ignore metrics whose interval happens to share a factor with the log interval
            log_rate=log_rate,
        )
        if self.eval_env is not None:
            if self.config.metrics_log_count > 0:
                eval_log_rate = JaxRate.build(
                    closest_prime(
                        int(
                            self.config.total_env_steps
                            / self.config.eval_every
                            // self.config.metrics_log_count
                        )
                    )
                )
            else:
                eval_log_rate = JaxRate.never()

            eval_rng, reset_rng_eval = jax.random.split(new_rng)
            iteration_state_eval = _IterationState(
                eval_rng,
                self._reset_env(
                    self.eval_env,
                    reset_rng_eval,
                    initial_agent_state,
                    set_env_seed=True,
                ),
                initial_agent_state,
                log_rate=eval_log_rate,
            )
        else:
            iteration_state_eval = None

        return jax.tree.map(
            jnp.asarray, _MainState(iteration_state_train, iteration_state_eval)
        )

    def _handle_model_reset(
        self,
        iteration: jax.Array,
        algorithm_state: AlgorithmStateType,
        agent_state: Any,
    ) -> AlgorithmStateType:
        return algorithm_state

    def _train_loop_fn(
        self,
        iteration: jax.Array,
        state: _MainState,
        final_iteration: int,
        initial_agent_state: Any,
    ) -> _MainState:
        if state.iteration_state_eval is not None:
            new_iteration_state_eval = state.iteration_state_eval.replace(
                agent_state=state.iteration_state_train.agent_state
            )
            new_iteration_state_eval = jax.lax.cond(
                iteration % self.config.eval_every == 0,
                lambda s: self.iteration_fn(
                    iteration=iteration,
                    env=self.eval_env,
                    evaluation_mode=True,
                    data_logger=self.__eval_logger,
                    iteration_state=s,
                ),
                lambda s: s,
                new_iteration_state_eval,
            )
        else:
            new_iteration_state_eval = None

        def reset_model(iteration: jax.Array, state: _MainState) -> _MainState:
            jax.debug.print("Resetting model at iteration {itr}", itr=iteration)

            new_state = cursor(state).iteration_state_train.agent_state.set(
                initial_agent_state
            )

            return cursor(new_state).iteration_state_train.algorithm_state.set(
                self._handle_model_reset(
                    iteration,
                    new_state.iteration_state_train.algorithm_state,
                    initial_agent_state,
                )
            )

        state = jax.lax.cond(
            iteration % self.model_reset_interval_iterations == 0,
            reset_model,
            lambda i, s: s,
            iteration,
            state,
        )

        new_iteration_state_train = self.iteration_fn(
            iteration=iteration,
            env=self.train_env,
            evaluation_mode=False,
            data_logger=self.__train_logger,
            iteration_state=state.iteration_state_train,
        )
        new_state = _MainState(
            new_iteration_state_train,
            new_iteration_state_eval,
        )

        if self.config.save_checkpoints:
            checkpoint_iterations = jnp.round(
                jnp.linspace(
                    0,
                    self.total_iterations - 1,
                    num=self.config.intermediate_checkpoint_count + 2,
                )[1:]
            ).astype(jnp.int32)

            new_state = jax.lax.cond(
                jnp.any(iteration == checkpoint_iterations)
                | (iteration == final_iteration),
                self._save_checkpoint,
                lambda i, s: s,
                iteration,
                new_state,
            )

        return new_state

    # JITing this function prevents the replay buffer from being needlessly copied
    @partial(
        jax.jit,
        static_argnames=(
            "self",
            "return_replay_buffer",
        ),
    )
    def _train(self, rng: jax.Array, return_replay_buffer: bool = False):
        initial_state = self._mk_initial_state(rng)
        state = initial_state
        existing_checkpoints = self.__checkpoint_manager.all_steps()
        if len(existing_checkpoints) > 0:
            prev_iteration = max(existing_checkpoints)
            state = self._load_checkpoint(
                prev_iteration,
                state,
            )
            initial_iteration = prev_iteration + 1
        else:
            initial_iteration = 0

        if self.config.env_steps is not None:
            end_iteration = min(
                self.total_iterations,
                initial_iteration
                + self.config.env_steps // self.env_steps_per_iteration,
            )
        else:
            end_iteration = self.total_iterations

        final_state: _MainState = jax.lax.fori_loop(
            initial_iteration,
            end_iteration,
            partial(
                self._train_loop_fn,
                final_iteration=end_iteration - 1,
                initial_agent_state=initial_state.iteration_state_train.agent_state,
            ),
            state,
        )

        # Careful: returning the replay buffer here will cause JAX to hold it twice in memory
        if return_replay_buffer:
            replay_buffer = (
                final_state.iteration_state_train.prev_step_state.trajectory_buffer
            )
        else:
            replay_buffer = None

        return TrainingOutput(
            final_state.iteration_state_train.agent_state, replay_buffer
        )

    def _eval_loop_fn(
        self,
        iteration: jax.Array,
        state: _MainState,
    ) -> _MainState:
        if state.iteration_state_eval is not None:
            new_iteration_state_eval = state.iteration_state_eval.replace(
                agent_state=state.iteration_state_train.agent_state
            )
            new_iteration_state_eval = jax.lax.cond(
                iteration % self.config.eval_every == 0,
                lambda s: self.iteration_fn(
                    iteration=iteration,
                    env=self.eval_env,
                    evaluation_mode=True,
                    data_logger=self.__eval_logger,
                    iteration_state=s,
                ),
                lambda s: s,
                new_iteration_state_eval,
            )
        else:
            new_iteration_state_eval = None
        new_iteration_state_train = self.iteration_fn(
            iteration=iteration,
            env=self.train_env,
            evaluation_mode=True,
            data_logger=self.__train_logger,
            iteration_state=state.iteration_state_train,
        )
        new_state = _MainState(
            new_iteration_state_train,
            new_iteration_state_eval,
        )

        return new_state

    def _evaluate_cp(self, iteration: jax.Array, state: _MainState, num_env_steps: int):
        state_load = cursor(
            state
        ).iteration_state_train.prev_step_state.trajectory_buffer.set(
            TrajectoryBuffer.build(
                self._get_trajectory_variable_spec(),
                size=1,
                stream_count=self.train_env.num_envs,
            )
        )
        state_load = self._load_checkpoint(
            iteration, state_load, retain_data_logger_step=True
        )
        state = cursor(
            state_load
        ).iteration_state_train.prev_step_state.trajectory_buffer.set(None)

        return jax.lax.fori_loop(
            0,
            int(np.ceil(num_env_steps / self.env_steps_per_iteration)),
            self._eval_loop_fn,
            state,
        )

    @partial(
        jax.jit,
        static_argnames=("self", "num_env_steps"),
    )
    def _evaluate(self, rng: jax.Array, num_env_steps: int):
        existing_checkpoints = jnp.array(self.__checkpoint_manager.all_steps())
        if len(existing_checkpoints) == 0:
            raise RuntimeError("No checkpoints found. Please run the training first.")
        state = self._mk_initial_state(rng, create_trajectory_buffer=False)
        self.__data_logger.write({}, 0)
        jax.lax.fori_loop(
            0,
            len(existing_checkpoints),
            lambda i, s: self._evaluate_cp(existing_checkpoints[i], s, num_env_steps),
            state,
        )

    def evaluate(
        self,
        rng: jax.Array,
        env: VectorEnv,
        data_logger: DataLogger,
        num_env_steps: int,
        eval_env: VectorEnv | None = None,
    ) -> TrainingOutput:

        with self.__setup(
            env,
            data_logger,
            eval_env=eval_env,
        ) as c_self:
            return c_self._evaluate(rng, num_env_steps=num_env_steps)

    def run(
        self,
        rng: jax.Array,
        env: VectorEnv,
        data_logger: DataLogger,
        eval_env: VectorEnv | None = None,
        return_replay_buffer: bool = False,
    ) -> TrainingOutput:

        with self.__setup(
            env,
            data_logger,
            eval_env=eval_env,
        ) as c_self:

            if c_self.config.profile:
                options = jax.profiler.ProfileOptions()
                options.python_tracer_level = 0
                options.host_tracer_level = 0
                profiler_dir = c_self.__data_logger.run_directory / "profiler"
                profiler_dir.mkdir()
                profiler = jax.profiler.trace(
                    profiler_dir,
                    create_perfetto_link=c_self.config.profile_create_perfetto_link,
                    profiler_options=options,
                )
            else:
                profiler = nullcontext()

            with profiler:
                return jax.block_until_ready(
                    c_self._train(rng, return_replay_buffer=return_replay_buffer),
                )

    def __setattr__(self, key, value):
        if (
            self.__context is None
            and hasattr(self, key)
            and isinstance(getattr(self, key), ContextVar)
        ):
            raise ContextNotInitializedError(
                f"Cannot set context variable '{key}' outside of context."
            )
        elif self.__context == "initialized" and key in self.__context_variables:
            raise ContextVariableReadOnlyError(
                f"Cannot set context variable '{key}' after initialization."
            )
        super().__setattr__(key, value)

    def __getattribute__(self, item):
        object = super().__getattribute__(item)

        if isinstance(object, ContextVar):
            if self.__context in ["initializing", "initialized"]:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{item}'"
                )
            else:
                raise ContextNotInitializedError(
                    f"Cannot access context variable '{item}' outside of context."
                )

        return object

    def __prepare_env(
        self,
        env: gym.vector.VectorEnv,
        log_wrapper_kwargs: dict[str, Any] | None = None,
    ) -> JaxWrapper:
        if log_wrapper_kwargs is None:
            log_wrapper_kwargs = {}
        env = DetectImageObsWrapper(env)
        env = ensure_active_perception_vector_env(env)
        if self.config.classification_binary_reward and isinstance(
            env.loss_fn, CrossEntropyLossFn
        ):
            env = ClassificationBinaryRewardVectorWrapper(env)
        if self.config.use_rendering_as_observation:
            env = AddRenderObservationVectorWrapper(env, target_img_dtype=np.uint8)
        if self.config.treat_ap_gym_env_as_gym_env:
            env = MaskActivePerceptionEnvVectorWrapper(env)
        if self.config.control_strategy == ControlStrategy.RANDOM:
            env = RandomActionWrapper(env)
        elif self.config.control_strategy == ControlStrategy.GRID_2D:
            env = Grid2DActionWrapper(env)
        elif self.config.control_strategy != ControlStrategy.POLICY:
            raise NotImplementedError(
                f"Control strategy {self.config.control_strategy.name} not implemented."
            )
        env = GymVectorWrapper32(env)
        env = CheckMaxStepsVectorWrapper(env, self.max_episode_steps)
        env = LogWrapper(env, **log_wrapper_kwargs)
        if self.config.replay_buffer_max_image_size is not None:
            max_img_size = (self.config.replay_buffer_max_image_size,) * 2
        else:
            max_img_size = None
        if self.config.replay_buffer_img_dtype == ReplayBufferImageDType.F32:
            img_target_dtype = jnp.float32
        elif self.config.replay_buffer_img_dtype == ReplayBufferImageDType.F16:
            img_target_dtype = jnp.float16
        elif self.config.replay_buffer_img_dtype == ReplayBufferImageDType.U8:
            img_target_dtype = jnp.uint8
        else:
            raise ValueError(f"Unknown dtype {self.config.replay_buffer_img_dtype}")
        return JaxWrapper(
            env, max_img_size=max_img_size, img_target_dtype=img_target_dtype
        )

    @property
    def name(self) -> str:
        return self.__name

    @property
    def config(self) -> ConfigType | None:
        return self.__config

    @property
    def total_iterations(self) -> int | None:
        if self.config is None:
            return None
        if self.algorithm_settings is None:
            return None
        return self.config.total_env_steps // self.env_steps_per_iteration

    @property
    def env_steps_per_iteration(self) -> int | None:
        if self.algorithm_settings is None:
            return None
        return (
            self.algorithm_settings.batch_env_steps_per_iteration
            * self.train_env.num_envs
        )

    @property
    def effective_total_env_steps(self) -> int | None:
        if self.total_iterations is None:
            return None
        return self.total_iterations * self.env_steps_per_iteration

    @property
    def learning_starts(self):
        return max(
            self.config.learning_starts, self.algorithm_settings.min_learning_starts
        )

    @property
    def model_reset_interval_iterations(self) -> int:
        return int(
            math.ceil(
                self.total_iterations / (self.config.num_intermediate_model_resets + 1)
            )
        )

    @property
    def effective_max_episode_steps(self) -> int:
        if not self.config.reset_env_between_iterations:
            return self.max_episode_steps
        return min(
            self.max_episode_steps,
            self.algorithm_settings.batch_env_steps_per_iteration,
        )

    @property
    def memory_horizon_with_context(self) -> int:
        return min(self.effective_max_episode_steps, 2 * self.memory_horizon - 1)

    @property
    def memory_horizon(self) -> int:
        if self.config.max_memory_horizon is None:
            return self.max_episode_steps
        return min(self.config.max_memory_horizon, self.max_episode_steps)
