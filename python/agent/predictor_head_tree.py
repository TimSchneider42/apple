from __future__ import annotations

import inspect
from functools import partial
from typing import Callable

import flax.linen as nn
import gymnasium
import jax
from flax.linen.linear import default_kernel_init

from gym_space_map import gym_space_map
from pytree import PyTree
from tree_wrapper import TreeWrapper


class PredictorHeadTree(TreeWrapper["PredictorHeadTree", nn.Module], nn.Module):
    @nn.compact
    def __call__(self, x: jax.Array) -> PyTree[jax.Array]:
        return jax.tree.map(lambda head: head(x), self.orig_tree)

    @classmethod
    def is_leaf(cls, x: PyTree[nn.Module]) -> bool:
        return isinstance(x, nn.Module) or inspect.isfunction(x)

    @classmethod
    def from_primitive_action_space(
        cls,
        action_space: gymnasium.Space,
        kernel_init: nn.initializers.Initializer = default_kernel_init,
    ) -> Callable[[jax.Array], jax.Array]:
        if isinstance(action_space, gymnasium.spaces.Box):
            return nn.DenseGeneral(action_space.shape, kernel_init=kernel_init)
        else:
            raise NotImplementedError(f"Unsupported action space {action_space}")

    @classmethod
    def from_action_space(
        cls,
        action_space: gymnasium.Space,
        kernel_init: nn.initializers.Initializer = default_kernel_init,
    ) -> PredictorHeadTree | Callable[[jax.Array], jax.Array]:
        return PredictorHeadTree.wrap(
            gym_space_map(
                partial(
                    cls.from_primitive_action_space,
                    kernel_init=kernel_init,
                ),
                action_space,
            )
        )
