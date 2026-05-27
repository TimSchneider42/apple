from __future__ import annotations

from abc import abstractmethod
from functools import partial
from typing import Union, Sequence, overload, TypeVar, Generic

import flax.struct
import jax
import jax.numpy as jnp

from pytree import PyTree
from ring_buffer import RingBuffer

TrajectoryType = TypeVar("TrajectoryType", bound=PyTree[jax.Array])


@flax.struct.dataclass
class TrajectoryBufferData(Generic[TrajectoryType]):
    variables: TrajectoryType
    start: jax.Array
    end: jax.Array
    step_no: jax.Array
    trajectory_id: jax.Array


@flax.struct.dataclass
class TrajectoryBatch(Generic[TrajectoryType]):
    data: TrajectoryBufferData[TrajectoryType]
    data_valid: jax.Array


@flax.struct.dataclass
class _TrajectoryBufferCache:
    trajectory_counts: dict[int, jax.Array]


@flax.struct.dataclass
class TrajectoryInfo:
    id: jax.Array
    start_index: jax.Array
    length: jax.Array


@flax.struct.dataclass
class FullTrajectoryInfo:
    start_index: jax.Array
    length: jax.Array
    stream_index: jax.Array


@flax.struct.dataclass
class TrajectoryBuffer(Sequence[TrajectoryType]):
    data: TrajectoryBufferData[TrajectoryType]
    start_index: jax.Array
    length: jax.Array
    trajectory_info: RingBuffer[TrajectoryInfo]

    _cache: _TrajectoryBufferCache | None = None

    @classmethod
    def build(
        cls, variables: TrajectoryType, stream_count: int = 1, capacity: int = 50_000
    ) -> TrajectoryBuffer[TrajectoryType]:
        shape = (stream_count, capacity)

        # It is important to zero-initialize the data as it might end up in a batch norm layer somewhere (through the
        # context, which might be out of bounds)
        data = jax.tree.map(
            lambda vs: jnp.zeros(shape + vs.shape, dtype=vs.dtype), variables
        )
        start = jnp.zeros(shape, dtype=jnp.bool_)
        end = jnp.zeros(shape, dtype=jnp.bool_)
        step_no = jnp.zeros(shape, dtype=jnp.int32)
        traj_id = jnp.zeros(shape, dtype=jnp.int32)
        return TrajectoryBuffer(
            TrajectoryBufferData(data, start, end, step_no, traj_id),
            jnp.array(0),
            jnp.array(0),
            cls.__create_trajectory_info_buffer(stream_count, capacity + 1),
        )

    @staticmethod
    def __create_trajectory_info_buffer(
        stream_count: int, capacity: int
    ) -> RingBuffer[TrajectoryInfo]:
        return jax.vmap(
            lambda _: RingBuffer.build(
                TrajectoryInfo(
                    jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
                    jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
                    jax.ShapeDtypeStruct(shape=(), dtype=jnp.int32),
                ),
                batch_shape=(),
                capacity=capacity,
            )
        )(jnp.arange(stream_count))

    @classmethod
    def from_data(
        cls,
        data: TrajectoryBufferData[TrajectoryType],
        index: jax.Array | int = 0,
        length: jax.Array | int | None = None,
    ) -> TrajectoryBuffer[TrajectoryType]:
        if length is None:
            length = data.start.shape[-1]
        length = jnp.asarray(length, dtype=jnp.int32)
        index = jnp.asarray(index, dtype=jnp.int32)
        buffer = TrajectoryBuffer(
            data,
            index,
            length,
            cls.__create_trajectory_info_buffer(
                data.start.shape[0], data.start.shape[-1] + 1
            ),
        )
        return buffer.rebuild_trajectory_info_buffer()

    @partial(jax.jit, donate_argnums=(0,))
    def add_step(
        self, done: jax.Array, variables: TrajectoryType
    ) -> TrajectoryBuffer[TrajectoryType]:
        """
        Adds a batch of single steps to the buffer. The step is added to the trajectory that is currently being
        recorded.
        :param done:        A boolean array of shape (batch_size,) that indicates whether the current trajectory is
                            done. The variables passed along with this flag will be stored as the last step of the
                            trajectory.
        :param variables:   A PyTree containing the variables to be stored.
        :return:
        """
        target_index = (self.start_index + self.length) % self.capacity_steps
        first_step = self.length == 0
        # Note that it is fine to ignore the case self.length == 0 here, because in that case first_step is True anyway
        prev_idx = (target_index - 1) % self.capacity_steps
        prev_traj_ended = self.data.end[:, prev_idx]
        prev_step_no = self.data.step_no[:, prev_idx]
        is_start = first_step | prev_traj_ended

        # The old trajectory start flags are only valid if the buffer was previously full
        old_is_start = self.data.start[:, target_index] & (
            self.length == self.capacity_steps
        )

        new_length = jnp.minimum(self.length + 1, self.capacity_steps)
        new_index = (target_index - new_length + 1) % self.capacity_steps

        last_ids = jax.vmap(lambda b: b[-1])(self.trajectory_info).id
        current_ids = last_ids * ~first_step - 1 * first_step

        traj_ids = jnp.where(
            is_start, jnp.max(current_ids) + jnp.cumsum(is_start), last_ids
        )

        # This looks terribly inefficient but is actually much better than doing this in batch. The reason is that
        # doing it in batch causes XLA to transpose the entire buffer to be able to do a continuous write and then
        # transpose it back again.
        def update_stream(
            i: jax.Array | int,
            carry: tuple[
                TrajectoryBufferData[TrajectoryType], RingBuffer[TrajectoryInfo]
            ],
        ):
            data, info_buffer = carry
            new_vars = jax.tree.map(
                lambda d_old, d_new: d_old.at[i, target_index].set(d_new[i]),
                data.variables,
                variables,
            )

            new_start = data.start.at[i, target_index].set(is_start[i])
            new_step_no = data.step_no.at[i, target_index].set(
                (prev_step_no[i] + 1) * ~is_start[i]
            )
            new_end = data.end.at[i, target_index].set(done[i])
            new_traj_ids = data.trajectory_id.at[i, target_index].set(traj_ids[i])

            new_data = TrajectoryBufferData(
                new_vars, new_start, new_end, new_step_no, new_traj_ids
            )

            # When adding a new start, we need to add it to the ring buffer
            # For whatever reason, conditionally adding to the ring buffer causes a copy of the replay buffer to be
            # made. We avoid it here.
            info_target_index = (
                info_buffer.index[i]
                + info_buffer.length[i]
                + is_start[i].astype(jnp.int32)
                - 1
            ) % info_buffer.capacity

            current_trajectory_info = jax.lax.cond(
                is_start[i],
                lambda: TrajectoryInfo(
                    traj_ids[i], target_index, jnp.zeros((), dtype=jnp.int32)
                ),
                lambda: jax.tree.map(
                    lambda d: d[i, info_target_index], info_buffer.data
                ),
            )
            current_trajectory_info = current_trajectory_info.replace(
                length=current_trajectory_info.length + 1
            )

            new_info_data = jax.tree.map(
                lambda d_old, d_new: d_old.at[i, info_target_index].set(d_new),
                info_buffer.data,
                current_trajectory_info,
            )

            # When overwriting an old start, we need to pop it from the ring buffer
            new_info_length_i = jnp.minimum(
                info_buffer.length[i]
                - old_is_start[i].astype(jnp.int32)
                + is_start[i].astype(jnp.int32),
                info_buffer.capacity,
            )
            new_info_index_i = (
                info_target_index - new_info_length_i + 1
            ) % info_buffer.capacity

            new_info_buffer = info_buffer.fork(
                new_info_data,
                info_buffer.index.at[i].set(new_info_index_i),
                info_buffer.length.at[i].set(new_info_length_i),
            )
            return new_data, new_info_buffer

        new_data, new_info_buffer = jax.lax.fori_loop(
            0, self.batch_size, update_stream, (self.data, self.trajectory_info)
        )

        return TrajectoryBuffer(
            new_data,
            new_index,
            new_length,
            new_info_buffer,
        )

    @partial(jax.jit, donate_argnums=(0,))
    def rebuild_trajectory_info_buffer(self) -> TrajectoryBuffer[TrajectoryType]:
        start_aligned = jnp.roll(self.data.start, -self.start_index, axis=1)
        steps = jnp.arange(self.capacity_steps)[None, :]
        valid = steps < self.length
        fill_value = jnp.iinfo(jnp.int32).max
        traj_start_idx_aligned = jax.vmap(
            lambda s: jnp.where(
                s,
                size=self.capacity_steps + 2,
                fill_value=fill_value,
            )[0]
        )(start_aligned & valid)
        num_trajectories = jax.vmap(
            lambda si: jnp.searchsorted(si, fill_value, side="left")
        )(traj_start_idx_aligned)
        traj_lengths = traj_start_idx_aligned[:, 1:] - traj_start_idx_aligned[:, :-1]
        traj_lengths = traj_lengths.at[
            jnp.arange(traj_lengths.shape[0]), num_trajectories - 1
        ].set(
            self.capacity_steps
            - traj_start_idx_aligned[
                jnp.arange(traj_lengths.shape[0]), num_trajectories - 1
            ]
        )
        traj_start_idx = traj_start_idx_aligned[:, :-1] + self.start_index
        ids = self.data.trajectory_id[
            jnp.arange(traj_start_idx.shape[0])[:, None], traj_start_idx
        ]

        return TrajectoryBuffer(
            self.data,
            self.start_index,
            self.length,
            jax.vmap(
                lambda s, l, t, i: RingBuffer(
                    TrajectoryInfo(i, s, l),
                    batch_shape=(),
                    capacity=self.capacity_steps + 1,
                    index=jnp.array(0),
                    length=t,
                )
            )(traj_start_idx, traj_lengths, num_trajectories, ids),
        )

    @partial(jax.jit, donate_argnums=(0,))
    def replace_last_step(
        self, done: jax.Array, variables: TrajectoryType
    ) -> TrajectoryBuffer[TrajectoryType]:
        actual_index = (self.start_index - 1) % self.capacity_steps
        new_variables = jax.tree.map(
            lambda d_old, d_new: (
                d_old if d_new is None else d_old.at[:, actual_index].set(d_new)
            ),
            self.data.variables,
            variables,
        )
        return TrajectoryBuffer(
            TrajectoryBufferData(
                new_variables,
                self.data.start,
                self.data.end.at[:, actual_index].set(done),
                self.data.step_no,
                self.data.trajectory_id,
            ),
            self.start_index,
            self.length,
            self.trajectory_info,
        )

    @partial(jax.jit, donate_argnums=(0,), static_argnames=("sequence_length",))
    def populate_cache(
        self,
        sequence_length: int,
    ) -> TrajectoryBuffer[TrajectoryType]:
        current_cache = {} if self._cache is None else self._cache.trajectory_counts
        if sequence_length in current_cache:
            return self

        # Ensure that the trajectory starts at least sequence_length steps before the end of the data
        idx = jnp.arange(-sequence_length, 0)
        last_trajectories = jax.vmap(lambda b: b[idx])(self.trajectory_info)
        last_trajectories_valid = idx >= -self.trajectory_info.length[:, None]
        start_indices_norm = (
            last_trajectories.start_index - self.start_index
        ) % self.capacity_steps
        too_close_to_end = self.length - start_indices_norm < sequence_length
        too_close_to_end_counts = jnp.sum(
            too_close_to_end & last_trajectories_valid, axis=1
        )

        return TrajectoryBuffer(
            self.data,
            self.start_index,
            self.length,
            self.trajectory_info,
            _cache=_TrajectoryBufferCache(
                trajectory_counts={
                    **current_cache,
                    sequence_length: self.trajectory_info.length
                    - too_close_to_end_counts,
                }
            ),
        )

    @partial(
        jax.jit, static_argnames=("batch_size", "sequence_length", "context_length")
    )
    def sample_batch(
        self,
        rng: jax.Array,
        batch_size: int,
        sequence_length: int,
        context_length: int = 0,
        active_stream_mask: jax.Array | None = None,
        oversampling_factor: int = 16,
    ) -> TrajectoryBatch[TrajectoryType]:
        # The following must hold for valid starting points:
        # 1. There must be enough space until the end of the buffer (ensured when computing the trajectory counts).
        # 2. One of the following must hold
        #    a) The step is a valid start step.
        #    b) The step is at least sequence_length steps before the end of the episode. This allows us to use shorter
        #       context windows at training time in many cases.
        #
        rng_trajectory, rng_subsample, rng_offset = jax.random.split(rng, 3)
        self_populated = self.populate_cache(sequence_length)
        trajectory_counts = self_populated._cache.trajectory_counts[sequence_length]
        if active_stream_mask is None:
            active_stream_mask = jnp.ones(trajectory_counts.shape[0], dtype=jnp.bool_)
        start_indices = jax.random.randint(
            rng_trajectory,
            (batch_size * oversampling_factor,),
            0,
            jnp.sum(trajectory_counts * active_stream_mask),
        )
        trajectory_info = self.__get_trajectory_info(
            start_indices, trajectory_counts * active_stream_mask
        )
        if oversampling_factor != 1:
            # This is essentially a trade-off between performance and sampling uniformity. Ideally, every step should
            # have the same probability of being drawn. However, here we sample trajectories, meaning that steps from
            # shorter trajectories get preferred. To counteract this while avoiding the cost of a large categorical
            # sample, we sample twice: once we pick oversampling_factor * batch_size trajectories, and then we sample
            # categorically from them based on their length.
            transition_count = trajectory_info.length - 1
            selected_indices = jax.random.choice(
                rng_subsample,
                jnp.arange(len(trajectory_info.length)),
                shape=(batch_size,),
                replace=False,
                p=transition_count,
            )
            trajectory_info = FullTrajectoryInfo(
                trajectory_info.start_index[selected_indices],
                trajectory_info.length[selected_indices],
                trajectory_info.stream_index[selected_indices],
            )
        max_offset = trajectory_info.length - sequence_length
        offsets = jax.random.randint(rng_offset, (batch_size,), 0, max_offset + 1)
        actual_indices = (
            (trajectory_info.start_index + offsets - context_length)[:, None]
            + jnp.arange(sequence_length + context_length)
        ) % self.capacity_steps

        batch_data = jax.tree.map(
            lambda d: d[trajectory_info.stream_index[:, None], actual_indices],
            self.data,
        )
        indices_norm = (actual_indices - self.start_index) % self.capacity_steps
        data_valid = (indices_norm >= 0) & (indices_norm < self.length)
        return TrajectoryBatch(batch_data, data_valid)

    def clear(self) -> TrajectoryBuffer[TrajectoryType]:
        return TrajectoryBuffer(
            self.data,
            jnp.array(0),
            jnp.array(0),
            jax.vmap(lambda b: b.clear())(self.trajectory_info),
        )

    def get_trajectory(
        self, index: int, ensure_complete: bool = True
    ) -> TrajectoryBufferData[TrajectoryType]:
        return self.get_trajectories([index], ensure_complete)[0]

    def get_trajectories(
        self, index: Union[slice, Sequence[int]], ensure_complete: bool = True
    ):
        """
        Returns a single trajectory or a list of trajectories from the buffer. Only complete trajectories are returned.
        Note: This implementation is non-jittable and not very efficient. It is only meant for testing purposes.
        :param index:           Index, slice, or list of indices that indicates which trajectories to return.
        :param ensure_complete: If True, only complete trajectories are returned. If False, incomplete trajectories are
                                returned as well.
        :return: A single trajectory or a list of trajectories.
        """
        if ensure_complete:
            count = self.complete_trajectory_count
        else:
            count = self.trajectory_count

        if isinstance(index, slice):
            seq = index.indices(count)
        else:
            seq = index

        output = []
        for i in seq:
            if i < 0 or i >= count:
                raise IndexError(
                    f"Index {i} is out of bounds for buffer of size {count}."
                )
            if ensure_complete:
                trajectory_info = self.get_complete_trajectory_info(i)
            else:
                trajectory_info = self.get_trajectory_info(i)

            indices = (
                trajectory_info.start_index + jnp.arange(trajectory_info.length)
            ) % self.capacity_steps
            output.append(
                jax.tree.map(
                    lambda d: d[trajectory_info.stream_index, indices], self.data
                )
            )

        return output

    def get_complete_trajectory_info(
        self, trajectory_index: jax.Array | int
    ) -> FullTrajectoryInfo:
        return self.__get_trajectory_info(
            trajectory_index, self.complete_trajectory_counts_per_stream
        )

    def get_trajectory_info(
        self, trajectory_index: jax.Array | int
    ) -> FullTrajectoryInfo:
        return self.__get_trajectory_info(
            trajectory_index, self.trajectory_counts_per_stream
        )

    def __get_trajectory_info(
        self, trajectory_index: jax.Array | int, trajectory_counts_per_stream: jax.Array
    ) -> FullTrajectoryInfo:
        trajectory_counts_cum = jnp.cumsum(trajectory_counts_per_stream)
        stream_indices = jnp.searchsorted(
            trajectory_counts_cum, trajectory_index, side="right"
        )
        inner_stream_start_indices = (
            trajectory_index
            - jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), trajectory_counts_cum])[
                stream_indices
            ]
        )
        trajectory_info = jax.tree.map(
            lambda d: d[
                stream_indices,
                self.trajectory_info.index[stream_indices] + inner_stream_start_indices,
            ],
            self.trajectory_info.data,
        )
        return FullTrajectoryInfo(
            trajectory_info.start_index, trajectory_info.length, stream_indices
        )

    @overload
    @abstractmethod
    def __getitem__(self, index: int) -> TrajectoryBufferData[TrajectoryType]: ...

    @overload
    @abstractmethod
    def __getitem__(
        self, index: Union[slice, Sequence[int]]
    ) -> Sequence[TrajectoryBufferData[TrajectoryType]]: ...

    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        return jax.tree.map(
            lambda d: d[:, (self.start_index + index) % self.capacity_steps], self.data
        )

    def __len__(self):
        return self.step_count

    @property
    def step_count(self) -> jax.Array:
        return self.length

    @property
    def complete_trajectory_counts_per_stream(self):
        current_trajectory_complete = self.data.end[
            :, (self.start_index + self.length - 1) % self.capacity_steps
        ]
        return self.trajectory_counts_per_stream - (
            ~current_trajectory_complete
        ).astype(jnp.int32)

    @property
    def complete_trajectory_count(self):
        return jnp.sum(self.complete_trajectory_counts_per_stream)

    @property
    def trajectory_counts_per_stream(self) -> jax.Array:
        return self.trajectory_info.length

    @property
    def trajectory_count(self) -> jax.Array:
        return jnp.sum(self.trajectory_counts_per_stream)

    @property
    def capacity_steps(self) -> int:
        return self.data.start.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.data.start.shape[0]

    @property
    def data_aligned(self) -> TrajectoryBufferData[TrajectoryType]:
        return jax.tree.map(lambda d: jnp.roll(d, -self.start_index, axis=1), self.data)

    @property
    def valid_data(self) -> TrajectoryBufferData[TrajectoryType]:
        return jax.tree.map(lambda d: d[:, : self.length], self.data_aligned)
