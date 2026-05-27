from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import gymnasium as gym
import numpy as np

from algorithm.gym_space_embedding import (
    get_default_embedding_factories,
)
from ap_gym import ImageSpace
from metric_log_level import MetricLogLevel
from models import (
    ImageScaler,
    ChannelBroadcaster,
    InputEncoderSequence,
    VisionTransformer,
    ChannelAggregator,
    ImageConverter,
    Normalization,
    NORMALIZATION_FACTORIES,
    InputEncoder,
    ViTEmbeddings,
)
from .transformer_config import TransformerConfig


class ImageFormat(str, Enum):
    AS_IS = "as_is"
    RGB = "rgb"
    GRAYSCALE = "grayscale"


@dataclass
class ImageEncoderConfig(ABC):
    # Target image size for the image encoder.
    target_image_size: int

    # Target format for the image encoder.
    target_image_format: ImageFormat

    @abstractmethod
    def _make_encoder(
        self,
        space: ImageSpace,
        network_probe_level: MetricLogLevel = MetricLogLevel.BASIC,
    ) -> InputEncoder:
        pass

    def make_image_encoder(
        self,
        space: ImageSpace,
        network_probe_level: MetricLogLevel = MetricLogLevel.BASIC,
    ) -> InputEncoder:
        _target_image_size = (
            space.shape[0] if self.target_image_size <= 0 else self.target_image_size
        )
        if self.target_image_format == ImageFormat.AS_IS:
            _target_channels = space.shape[-1]
        elif self.target_image_format == ImageFormat.RGB:
            _target_channels = 3
        elif self.target_image_format == ImageFormat.GRAYSCALE:
            _target_channels = 1
        else:
            raise ValueError(
                f"Unknown target image channel type '{self.target_image_format}'."
            )
        low = np.min(space.low, axis=(-2, -3), keepdims=True)
        high = np.min(space.high, axis=(-2, -3), keepdims=True)
        assert np.all(low == np.max(space.low, axis=(-2, -3)))
        assert np.all(high == np.max(space.high, axis=(-2, -3)))
        shape = space.batch_shape + (
            _target_image_size,
            _target_image_size,
            _target_channels,
        )
        low = np.broadcast_to(low, shape)
        high = np.broadcast_to(high, shape)
        new_space = ImageSpace(
            _target_image_size,
            _target_image_size,
            _target_channels,
            space.batch_shape,
            space.dtype,
            low=low,
            high=high,
        )
        encoder = self._make_encoder(new_space, network_probe_level=network_probe_level)
        encoder_lst = [encoder]
        if self.target_image_size > 0:
            encoder_lst = [
                ImageScaler((_target_image_size, _target_image_size))
            ] + encoder_lst
        if _target_channels != space.shape[-1]:
            if _target_channels == 3:
                assert space.shape[-1] == 1
                encoder_lst = [ChannelBroadcaster()] + encoder_lst
            else:
                encoder_lst = [ChannelAggregator()] + encoder_lst
        encoder_lst = [ImageConverter(space.low, space.high)] + encoder_lst
        return InputEncoderSequence(encoder_lst)


@dataclass
class DenseImageEncoderConfig(ImageEncoderConfig):
    _type: str = "DenseImageEncoderConfig"  # Do not override

    def _make_encoder(
        self,
        space: ImageSpace,
        network_probe_level: MetricLogLevel = MetricLogLevel.BASIC,
    ) -> InputEncoder:
        return get_default_embedding_factories()[gym.spaces.Box](space)


@dataclass
class ViTImageEncoderConfig(ImageEncoderConfig):
    # Patch size for the ViT model.
    patch_size: int

    # Add a final layer norm in the ViT model.
    add_final_layer_norm: bool

    # Number of hidden dimensions in the image encoder.
    embedding_dims: int

    # Transformer part of the ViT model
    transformer: TransformerConfig

    # Normalization to use after patch embedding.
    patch_normalization: Normalization

    # Whether to use absolute positional embeddings
    use_abs_positional_embeddings: bool

    _type: str = "ViTImageEncoderConfig"  # Do not override

    def _make_encoder(
        self,
        space: ImageSpace,
        network_probe_level: MetricLogLevel = MetricLogLevel.BASIC,
    ) -> InputEncoder:
        num_patches_w = space.width // self.patch_size
        num_patches_h = space.height // self.patch_size
        num_patches = num_patches_w * num_patches_h + 1
        return VisionTransformer(
            embedding=ViTEmbeddings(
                self.embedding_dims,
                self.patch_size,
                patch_normalization=NORMALIZATION_FACTORIES[self.patch_normalization](),
                use_abs_positional_embeddings=self.use_abs_positional_embeddings,
            ),
            encoder=self.transformer.mk_transformer_encoder(
                min_sequence_length=num_patches,
            ),
            add_final_layer_norm=self.add_final_layer_norm,
        )
