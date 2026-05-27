from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

from .multi_modal_sequence_embedding import InputEncoder
from .normalization import BaseNormalization, NoNorm
from .transformer import TransformerEncoder
from .util import maybe_add_dims


class ViTPatchEmbeddings(nn.Module):
    embedding_size: int = 768
    patch_size: int = 16
    initializer_range: float = 0.02
    normalization: BaseNormalization = NoNorm()

    @nn.compact
    def __call__(
        self,
        pixel_values: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ):
        if (
            pixel_values.shape[-2] % self.patch_size != 0
            or pixel_values.shape[-3] % self.patch_size != 0
        ):
            raise ValueError(
                f"Image dimensions must be divisible by the patch size. "
                f"Got image size {pixel_values.shape[-3:-1]} and patch size {self.patch_size}."
            )

        bs, h, w, c = pixel_values.shape
        num_patches_w = pixel_values.shape[-2] // self.patch_size
        num_patches_h = pixel_values.shape[-3] // self.patch_size

        pos = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(-1, 1, num_patches_h),
                jnp.linspace(-1, 1, num_patches_w),
                indexing="ij",
            ),
            axis=-1,
        )
        pos_flat = jnp.reshape(pos, (num_patches_h * num_patches_w, 2))

        pixel_values_reshaped = (
            pixel_values.reshape(
                (bs, num_patches_h, self.patch_size, num_patches_w, self.patch_size, c)
            )
            .transpose(0, 1, 3, 2, 4, 5)
            .reshape(
                (
                    bs,
                    num_patches_h * num_patches_w,
                    self.patch_size * self.patch_size * c,
                )
            )
        )

        embeddings = nn.Dense(
            self.embedding_size,
            kernel_init=jax.nn.initializers.variance_scaling(
                self.initializer_range**2, "fan_in", "truncated_normal"
            ),
        )(pixel_values_reshaped)
        embeddings = self.normalization(embeddings)
        return embeddings, pos_flat


class ViTEmbeddings(nn.Module):
    embedding_size: int = 768
    patch_size: int = 16
    initializer_range: float = 0.02
    dropout_prob: float = 0.0
    patch_normalization: BaseNormalization = NoNorm()
    use_abs_positional_embeddings: bool = True

    @nn.compact
    def __call__(
        self,
        pixel_values: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        embeddings, patch_pos = ViTPatchEmbeddings(
            embedding_size=self.embedding_size,
            patch_size=self.patch_size,
            initializer_range=self.initializer_range,
            normalization=self.patch_normalization,
        )(pixel_values, evaluation_mode=evaluation_mode, norm_mask=norm_mask)

        cls_token = self.param(
            "cls_token",
            jax.nn.initializers.variance_scaling(
                self.initializer_range**2, "fan_in", "truncated_normal"
            ),
            (1, 1, self.embedding_size),
        )
        cls_tokens = jnp.broadcast_to(
            cls_token, (pixel_values.shape[0], 1, self.embedding_size)
        )

        num_patches, channels = patch_pos.shape
        embeddings = jnp.concatenate((cls_tokens, embeddings), axis=1)
        pos = jnp.concatenate((jnp.zeros((1, channels)), patch_pos), axis=-2)

        if self.use_abs_positional_embeddings:
            position_embeddings = self.param(
                "position_embeddings",
                jax.nn.initializers.variance_scaling(
                    self.initializer_range**2, "fan_in", "truncated_normal"
                ),
                (1, embeddings.shape[-2], self.embedding_size),
            )
            embeddings += position_embeddings

        embeddings = nn.Dropout(rate=self.dropout_prob)(
            embeddings, deterministic=evaluation_mode
        )
        return embeddings, pos


class VisionTransformer(nn.Module, InputEncoder):
    embedding: ViTEmbeddings
    encoder: TransformerEncoder
    add_final_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        *,
        evaluation_mode: bool = False,
        norm_mask: jax.Array | None = None,
    ) -> jax.Array:
        embedding, pos = self.embedding(
            x, evaluation_mode=evaluation_mode, norm_mask=norm_mask
        )

        encoder_output = self.encoder(
            embedding,
            src_pos=pos[None],
            evaluation_mode=evaluation_mode,
            norm_mask=maybe_add_dims(norm_mask, 1),
        )

        if self.add_final_layer_norm:
            encoder_output = nn.LayerNorm(epsilon=1e-12)(
                encoder_output, norm_mask=maybe_add_dims(norm_mask, 2)
            )

        return encoder_output[..., 0, :]
