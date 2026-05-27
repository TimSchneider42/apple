from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Type

import flax.linen as nn
import gymnasium as gym
import numpy as np
from flax.linen.linear import default_kernel_init
from gymnasium.vector import VectorEnv
from hydra.core.config_store import ConfigStore

from agent import (
    ActorHeadTree,
    SACConstrainedNormal,
    TruncatedNormal,
    ActObsTuple,
    PredictorHeadTree,
)
from algorithm.gym_space_embedding import (
    get_default_embedding_factories,
    EmbeddingFactory,
    embeddings_from_space,
)
from ap_gym import ImageSpace
from jax_util import get_act_fn
from metric_log_level import MetricLogLevel
from models import (
    RNNSequenceEncoder,
    SimpleSequenceEncoder,
    MultiModalSequenceEncoder,
    SequenceEncoder,
    MultiModalSequenceEmbedding,
    IdSequenceEncoder,
    Normalization,
    NORMALIZATION_FACTORIES,
    TransformerSequenceEncoder,
    BoundedDenseGeneralType,
    InputEncoderSequence,
    NormalizationInputEncoder,
    InputEncoder,
)
from .image_encoder_config import (
    ImageEncoderConfig,
    DenseImageEncoderConfig,
    ImageFormat,
    ViTImageEncoderConfig,
)
from .transformer_config import TransformerConfig


@dataclass
class ModelConfig:
    model_types: dict[str, str]
    model_instances: tuple[str, ...]

    @classmethod
    def parse(
        cls, config_tuple: tuple[str, str, str], expected_instance_count: int = 2
    ) -> "ModelConfig":
        model_types = {}
        model_instances = tuple(cs.strip() for cs in config_tuple)
        if len(model_instances) != expected_instance_count:
            raise ValueError(
                f"Expected {expected_instance_count} model instances, got {len(model_instances)}: {model_instances}."
            )
        for cs in model_instances:
            model_types[cs] = re.findall(r"^([a-z]+)[0-9]+$", cs)[0]
        return ModelConfig(model_types, model_instances)

    @staticmethod
    def __find_factory(
        model_factories: dict[str, Callable[[], Any]], model_type: str
    ) -> Callable[[], Any]:
        model_type = model_type.lower()
        candidates = [k for k in model_factories if k.lower().startswith(model_type)]
        if len(candidates) == 0:
            raise ValueError(
                f"No such model type '{model_type}'. Choose from {', '.join(model_factories.keys())}."
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous model type '{model_type}' (could be {', '.join(candidates)})."
            )
        return model_factories[candidates[0]]

    def build_models(
        self, model_factories: dict[str, Callable[[], Any]]
    ) -> tuple[Any, Any, Any]:
        assert not any(c.isnumeric() or c == "," for k in model_factories for c in k)
        models = {
            k: self.__find_factory(model_factories, v)()
            for k, v in self.model_types.items()
        }
        return tuple(models[k] for k in self.model_instances)


def generate_embeddings(
    observation_space: gym.Space,
    action_space: gym.Space | None = None,
    embedding_factories: Mapping[Type[gym.Space], EmbeddingFactory] | None = None,
) -> MultiModalSequenceEmbedding:
    if embedding_factories is None:
        embedding_factories = get_default_embedding_factories()
    obs_embedding = embeddings_from_space(observation_space, embedding_factories)
    act_embedding = (
        None
        if action_space is None
        else embeddings_from_space(action_space, embedding_factories)
    )
    return MultiModalSequenceEmbedding(
        ActObsTuple(action=act_embedding, observation=obs_embedding)
    )


class ActivationFunction(str, Enum):
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    SINUS = "sinus"
    TANH = "tanh"


class ConstrainedNormalType(str, Enum):
    NONE = "none"
    SAC = "sac"
    TRUNCATED = "truncated"


CONSTRAINED_NORMAL_CLASSES = {
    ConstrainedNormalType.NONE: None,
    ConstrainedNormalType.SAC: SACConstrainedNormal,
    ConstrainedNormalType.TRUNCATED: TruncatedNormal,
}


@dataclass
class AgentConfig:
    # Number of dimensions of the latent space of the RNN model.
    rnn_memory_dims: int

    # Hidden dimensions of the embedding network of the RNN model.
    rnn_embedding_hidden_dims: tuple[int, ...]

    # Activation function for neural networks.
    act_fn: ActivationFunction

    # Use orthogonal layer initialization.
    orthogonal_layer_init: bool

    # Configuration of the TransformerSequenceEncoder
    transformer: TransformerConfig

    # Which normalization to use for the inputs of the transformer.
    trans_input_normalization: Normalization

    # Size of the observation/action embeddings for the transformer.
    trans_embedding_dims: int

    # Number of layers of the SimpleSequenceEncoder.
    simple_layer_count: int

    # Number hidden dimensions of the SimpleSequenceEncoder.
    simple_hidden_dims: int

    # Which normalization to use in the SimpleSequenceEncoder.
    simple_norm: Normalization

    # Observe actions in the memory model.
    observe_actions: bool

    # Image encoder configuration.
    image_encoder: ImageEncoderConfig

    # Model configuration to use for policy, predictor, and value function.
    model_config: tuple[str, str, str]

    # Use log scale for the standard deviation of the normal distribution in actor heads.
    act_std_use_log_scale: bool

    # Allowed bounding functions for the standard deviations of the normal distributions in actor heads.
    act_std_allowed_bounding_fns: tuple[BoundedDenseGeneralType, ...]

    # Minimum standard deviation for the normal distribution in actor heads.
    act_log_std_min: float | None

    # Maximum standard deviation for the normal distribution in actor heads.
    act_log_std_max: float | None

    # Allowed bounding functions for the means of the normal distributions in actor heads.
    act_mean_allowed_bounding_fns: tuple[BoundedDenseGeneralType, ...]

    # Whether to bound the mean of the normal distribution in actor heads.
    act_bound_mean: bool

    # Which constrained normal distribution to use in the actor head.
    constrained_normal_type: ConstrainedNormalType

    # Minimum number of pixels in the image. If an image has less than this number of pixels, it will be treated as a
    # dense input.
    image_min_pixels: int

    # Normalization to apply to the inputs before embedding.
    input_normalization: Normalization

    # Whether to use sinusoidal positional encodings in the transformer encoder
    use_sinusoidal_pos_enc: bool

    # Whether to scale the transformer inputs by the inverse square root of the embedding dimension
    trans_scale_inputs: bool

    def get_embedding_factories(
        self, network_probe_level: MetricLogLevel = MetricLogLevel.BASIC
    ):
        embedding_factories = get_default_embedding_factories(
            input_normalization=NORMALIZATION_FACTORIES[self.input_normalization]
        )

        def mk_image_encoder(
            space: ImageSpace,
        ) -> InputEncoder:
            if space.width * space.height >= self.image_min_pixels:
                img_enc = self.image_encoder.make_image_encoder(
                    space, network_probe_level=network_probe_level
                )
            else:
                img_enc = DenseImageEncoderConfig(
                    target_image_size=-1,
                    target_image_format=ImageFormat.AS_IS,
                ).make_image_encoder(space)
            img_enc = InputEncoderSequence(
                [
                    NormalizationInputEncoder(
                        NORMALIZATION_FACTORIES[self.input_normalization](axis=-1)
                    ),
                    img_enc,
                ]
            )
            return img_enc

        embedding_factories[ImageSpace] = mk_image_encoder
        return embedding_factories

    def make_sequence_encoders(
        self,
        action_space: gym.spaces.Space,
        observation_space: gym.spaces.Space,
        memory_horizon: int,
        count: int = 2,
        max_episode_steps: int | None = None,
        network_probe_level: MetricLogLevel = MetricLogLevel.BASIC,
    ) -> tuple[MultiModalSequenceEncoder, ...]:
        if not self.observe_actions:
            action_space = None
        embedding_factories = self.get_embedding_factories(
            network_probe_level=network_probe_level
        )

        def mk_transformer_factory(
            can_see_future: bool,
        ) -> Callable[[], TransformerSequenceEncoder]:
            def mk_transformer():
                attention_ignore_memory_horizon = (
                    max_episode_steps is not None
                    and memory_horizon >= max_episode_steps
                )
                return TransformerSequenceEncoder(
                    transformer_encoder=self.transformer.mk_transformer_encoder(
                        min_sequence_length=min(memory_horizon, max_episode_steps),
                    ),
                    embedding_dims=self.trans_embedding_dims,
                    memory_horizon=memory_horizon,
                    can_see_future=can_see_future,
                    encoding_normalization=NORMALIZATION_FACTORIES[
                        self.trans_input_normalization
                    ],
                    attention_ignore_memory_horizon=attention_ignore_memory_horizon,
                    use_positional_encoding=self.use_sinusoidal_pos_enc,
                    max_episode_length=max_episode_steps + 1,
                    scale_inputs=self.trans_scale_inputs,
                )

            return mk_transformer

        def mk_mm_sequence_encoder(
            sequence_encoder: SequenceEncoder,
        ) -> MultiModalSequenceEncoder:
            return MultiModalSequenceEncoder(
                embedding=generate_embeddings(
                    observation_space,
                    action_space,
                    embedding_factories=embedding_factories,
                ),
                sequence_encoder=sequence_encoder,
            )

        sequence_encoder_factories = {
            "gru": lambda: RNNSequenceEncoder(
                nn.GRUCell(features=self.rnn_memory_dims),
                embedding_net_hidden_units=self.rnn_embedding_hidden_dims,
            ),
            "lstm": lambda: RNNSequenceEncoder(
                nn.LSTMCell(features=self.rnn_memory_dims),
                embedding_net_hidden_units=self.rnn_embedding_hidden_dims,
            ),
            "transformer": mk_transformer_factory(can_see_future=False),
            "oracle_transformer": mk_transformer_factory(can_see_future=True),
            "simple": lambda: SimpleSequenceEncoder(
                act_fn=get_act_fn(self.act_fn),
                hidden_layers=self.simple_layer_count,
                hidden_dims=self.simple_hidden_dims,
                kernel_init=(
                    nn.initializers.orthogonal(np.sqrt(2))
                    if self.orthogonal_layer_init
                    else default_kernel_init
                ),
                normalization=NORMALIZATION_FACTORIES[self.simple_norm],
            ),
            "none": lambda: IdSequenceEncoder(),
        }
        model_factories = {
            k: lambda _v=v: mk_mm_sequence_encoder(_v())
            for k, v in sequence_encoder_factories.items()
        }
        model_config = ModelConfig.parse(
            self.model_config, expected_instance_count=count
        )
        return model_config.build_models(model_factories)

    def make_actor_head(self, vector_env: VectorEnv) -> ActorHeadTree:
        assert isinstance(vector_env.single_action_space, gym.spaces.Dict)
        return ActorHeadTree.from_action_space(
            vector_env.single_action_space["action"],
            std_allowed_bounding_fns=self.act_std_allowed_bounding_fns,
            std_use_log_scale=self.act_std_use_log_scale,
            log_std_min=self.act_log_std_min,
            log_std_max=self.act_log_std_max,
            mean_allowed_bounding_fns=self.act_mean_allowed_bounding_fns,
            act_bound_mean=self.act_bound_mean,
            kernel_init=(
                nn.initializers.orthogonal(0.01)
                if self.orthogonal_layer_init
                else default_kernel_init
            ),
            constrained_normal_type=CONSTRAINED_NORMAL_CLASSES[
                self.constrained_normal_type
            ],
        )

    def make_predictor_head(self, vector_env: VectorEnv) -> PredictorHeadTree:
        return PredictorHeadTree.from_action_space(
            vector_env.single_action_space["prediction"],
        )


cs = ConfigStore.instance()
cs.store(group="agent/image_encoder", name="dense_schema", node=DenseImageEncoderConfig)
cs.store(group="agent/image_encoder", name="vit_schema", node=ViTImageEncoderConfig)
