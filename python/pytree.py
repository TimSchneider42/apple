from __future__ import annotations

from typing import TypeVar, Union, Iterable, Mapping, Any

InnerType = TypeVar("InnerType")


PyTree = Union[
    InnerType,
    Iterable["NestedType[InnerType]"],
    Mapping[Any, "NestedType[InnerType]"],
    Any,
]
