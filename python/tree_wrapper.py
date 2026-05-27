from __future__ import annotations

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Union

import jax
from jax.tree_util import register_pytree_node_class

from pytree import PyTree

TSelf = TypeVar("TSelf", bound="TreeWrapper")
TLeaf = TypeVar("TLeaf")


@register_pytree_node_class
@dataclass
class TreeWrapper(ABC, Generic[TSelf, TLeaf]):
    inner: PyTree[Union[TLeaf, TSelf]]

    @classmethod
    def wrap(cls, pytree: PyTree[Union[TLeaf, Any]]) -> PyTree[TSelf]:
        if cls.is_leaf(pytree):
            return cls(pytree)
        return cls(jax.tree.map(cls.wrap, pytree, is_leaf=lambda x: x is not pytree))

    def __getattribute__(self, item):
        try:
            return object.__getattribute__(self, item)
        except AttributeError:
            return getattr(self.inner, item)

    def tree_flatten(self):
        children, inner_aux_data = jax.tree.flatten(self.inner)
        return children, (type(self), inner_aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        wrapper_type, inner_aux_data = aux_data
        return wrapper_type.wrap(jax.tree.unflatten(inner_aux_data, children))

    @property
    def orig_tree(self) -> PyTree[Union[TLeaf, Any]]:
        return jax.tree.map(
            lambda x: x.orig_tree if isinstance(x, type(self)) else x,
            self.inner,
            is_leaf=lambda x: x is not self.inner,
        )

    @classmethod
    @abstractmethod
    def is_leaf(cls, x: Union[TLeaf, Any]) -> bool:
        pass
