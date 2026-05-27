from __future__ import annotations

from functools import partial
from typing import Union, Sequence, TypeVar, Any

import flax.struct
import jax
import jax.numpy as jnp

from pytree import PyTree

DataType = TypeVar("DataType", bound=PyTree[jax.Array])


@flax.struct.dataclass
class RingBuffer(Sequence[DataType]):
    data: DataType
    batch_shape: tuple[int, ...] = flax.struct.field(pytree_node=False)
    capacity: int = flax.struct.field(pytree_node=False)
    index: jax.Array = flax.struct.field(default_factory=lambda: jnp.array(0))
    length: jax.Array = flax.struct.field(default_factory=lambda: jnp.array(0))

    @staticmethod
    def build(
        data_spec: DataType, batch_shape: tuple[int, ...] = (), capacity: int = 50_000
    ) -> "RingBuffer[DataType]":
        data = jax.tree.map(
            lambda vs: jnp.zeros(batch_shape + (capacity,) + vs.shape, dtype=vs.dtype),
            data_spec,
        )
        return RingBuffer(data, batch_shape, capacity)

    @partial(jax.jit, donate_argnums=(0,))
    def add(self, data: DataType) -> "RingBuffer[DataType]":
        data_flat = jax.tree.flatten(data)[0]
        if len(data_flat) == 0:
            return self
        step_count = data_flat[0].shape[self.step_dim]
        if step_count >= self.capacity:
            return self.fork(
                jax.tree.map(
                    lambda d: d[self._index_step(slice(-self.capacity, step_count))],
                    data,
                ),
                jnp.array(0),
                jnp.array(self.capacity),
            )
        else:
            start = self.index + self.length
            target_index = (start + jnp.arange(step_count)) % self.capacity
            new_data = jax.tree.map(
                lambda d_old, d_new: d_old.at[self._index_step(target_index)].set(
                    d_new
                ),
                self.data,
                data,
            )
            new_length = jnp.minimum(self.length + step_count, self.capacity)
            new_index = (target_index[-1] - new_length + 1) % self.capacity
            return self.fork(new_data, new_index, new_length)

    def resize(self, new_capacity: int) -> "RingBuffer[DataType]":
        if new_capacity == self.capacity:
            return self
        elif new_capacity < self.capacity:
            new_length = jnp.minimum(self.length, new_capacity)
            indices = (
                jnp.arange(new_capacity) + (self.index - new_capacity + self.length)
            ) % self.capacity
            return RingBuffer(
                jax.tree.map(
                    lambda x: x[(slice(None),) * self.step_dim + (indices,)],
                    self.data,
                ),
                batch_shape=self.batch_shape,
                capacity=new_capacity,
                index=jnp.array(0),
                length=new_length,
            )
        else:
            return RingBuffer(
                jax.tree.map(
                    lambda x: jnp.concatenate(
                        [
                            x,
                            jnp.zeros(
                                self.batch_shape
                                + (new_capacity - self.capacity,)
                                + x.shape[self.step_dim + 1 :],
                                dtype=x.dtype,
                            ),
                        ],
                        axis=self.step_dim,
                    ),
                    self.data_aligned,
                ),
                batch_shape=self.batch_shape,
                capacity=new_capacity,
                index=jnp.array(0),
                length=self.length,
            )

    @partial(jax.jit, donate_argnums=(0,))
    def replace_single(
        self, index: jax.Array | int, data: DataType
    ) -> "RingBuffer[DataType]":
        index = jnp.where(index < 0, index + self.length, index)
        new_data = jax.tree.map(
            lambda d, d_new: d.at[
                self._index_step((self.index + index) % self.capacity)
            ].set(d_new),
            self.data,
            data,
        )
        return self.fork(new_data, self.index, self.length)

    @partial(jax.jit, donate_argnums=(0,))
    def add_single(self, data: DataType) -> "RingBuffer[DataType]":
        return self.add(jax.tree.map(lambda d: d[self._index_step(None)], data))

    @partial(jax.jit, donate_argnums=(0,))
    def pop_front(self, count: int = 1) -> "RingBuffer[DataType]":
        return self.fork(
            self.data,
            (self.index + count) % self.capacity,
            jnp.maximum(self.length - count, 0),
        )

    def clear(self) -> "RingBuffer[DataType]":
        return self.fork(self.data, jnp.array(0), jnp.array(0))

    def _index_step(self, index: Any) -> tuple[Any, ...]:
        return (slice(None),) * self.step_dim + (index,)

    def fork(
        self, data: DataType, index: jax.Array, length: jax.Array
    ) -> "RingBuffer[DataType]":
        return RingBuffer(data, self.batch_shape, self.capacity, index, length)

    def __getitem__(self, index: Union[int, slice, Sequence[int]]) -> DataType:
        if isinstance(index, slice):
            index = index.indices(len(self))
        if not isinstance(index, int):
            index = jnp.asarray(index)
        index = jnp.where(index < 0, index + self.length, index)
        index = (index + self.index) % self.capacity
        return jax.tree.map(lambda d: d[self._index_step(index)], self.data)

    def __len__(self):
        return self.length

    @property
    def step_dim(self) -> int:
        return len(self.batch_shape)

    @property
    def data_aligned(self) -> DataType:
        return jax.tree.map(
            lambda d: jnp.roll(d, -self.index, axis=self.step_dim), self.data
        )

    @property
    def valid_data(self) -> DataType:
        return jax.tree.map(
            lambda d: d[self._index_step(slice(0, self.length))], self.data_aligned
        )
