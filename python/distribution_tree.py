from __future__ import annotations

from typing import Any, Union

import distrax
import jax

from pytree import PyTree
from tree_wrapper import TreeWrapper


class DistributionTree(
    TreeWrapper["DistributionTree", distrax.Distribution], distrax.Distribution
):
    def _sample_n(self, key: jax.Array, n: int) -> PyTree[distrax.Distribution]:
        tree_def = jax.tree.structure(
            self.orig_tree, is_leaf=lambda x: isinstance(x, distrax.Distribution)
        )
        keys = jax.tree.unflatten(tree_def, jax.random.split(key, tree_def.num_leaves))
        return jax.tree.map(
            lambda rng, dist: dist._sample_n(rng, n), keys, self.orig_tree
        )

    def _sample_n_and_log_prob(
        self, key: jax.Array, n: int
    ) -> PyTree[distrax.Distribution]:
        tree_def = jax.tree.structure(
            self.orig_tree, is_leaf=lambda x: isinstance(x, distrax.Distribution)
        )
        keys = jax.tree.unflatten(tree_def, jax.random.split(key, tree_def.num_leaves))
        if len(keys) == 0:
            return self.orig_tree, self.orig_tree
        return jax.tree.transpose(
            tree_def,
            jax.tree.structure(("*", "*")),
            jax.tree.map(
                lambda rng, dist: dist._sample_n_and_log_prob(rng, n),
                keys,
                self.orig_tree,
            ),
        )

    def log_prob(self, value: Any) -> PyTree[jax.Array]:
        return jax.tree.map(
            lambda dist, val: dist.log_prob(val),
            self.orig_tree,
            value,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )

    @property
    def event_shape(self) -> PyTree[tuple[int, ...]]:
        return jax.tree.map(
            lambda dist: dist.event_shape,
            self.orig_tree,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )

    def entropy(self) -> PyTree[jax.Array]:
        return jax.tree.map(
            lambda dist: dist.entropy(),
            self.orig_tree,
            is_leaf=lambda x: isinstance(x, distrax.Distribution),
        )

    def __getitem__(self, item):
        # Because distrax.Distribution defines it for some reason, we have to override it here.
        return self.inner[item]

    @classmethod
    def is_leaf(cls, x: Union[distrax.Distribution, Any]) -> bool:
        return isinstance(x, distrax.Distribution)
