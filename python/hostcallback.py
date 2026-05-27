from __future__ import annotations

import enum
import inspect
import logging
import math
import traceback
from abc import abstractmethod, ABC
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Callable, overload, Any, Protocol, Generic, TypeVar, ParamSpec

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct

from pytree import PyTree

logger = logging.getLogger(__name__)


def _check_direct_call(args, kwargs):
    if len(args) + len(kwargs) != 1:
        return False
    if len(args) == 1 and callable(args[0]):
        return args[0]
    elif "fn" in kwargs and callable(kwargs["fn"]):
        return kwargs["fn"]
    return False


PackingInfoType = tuple[tuple[tuple[tuple[int, ...], jnp.dtype], ...], Any]


def get_packing_info(tree: Any) -> tuple[ShapeDtypeStruct, PackingInfoType]:
    leaves, treedef = jax.tree_util.tree_flatten(tree)

    metadata = []
    byte_count = 0

    for leaf in leaves:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            leaf = jnp.asarray(leaf)
        metadata.append((leaf.shape, leaf.dtype))
        byte_count += leaf.dtype.itemsize * math.prod(leaf.shape)

    return ShapeDtypeStruct(shape=(byte_count,), dtype=jnp.uint8), (
        tuple(metadata),
        treedef,
    )


def pack_pytree(tree: Any) -> tuple[jax.Array, PackingInfoType]:
    """
    Flattens a pytree, converts all leaves to a single byte array,
    and returns the bytes plus metadata for reconstruction.
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)

    metadata = []
    byte_chunks = []

    for leaf in leaves:
        arr = jnp.asarray(leaf)
        metadata.append((arr.shape, arr.dtype))
        byte_chunks.append(arr.reshape(-1).view(jnp.uint8))

    if len(byte_chunks) > 0:
        output = jnp.concatenate(byte_chunks)
    else:
        output = jnp.array([], dtype=jnp.uint8)

    return output, (tuple(metadata), treedef)


@partial(jax.jit, static_argnums=[1])
def unpack_pytree(blob: jax.Array, packing_info: PackingInfoType) -> Any:
    """
    Takes a byte blob and metadata to reconstruct the original pytree.
    """
    metadata, treedef = packing_info
    leaves = deque()
    offset = 0

    for shape, dtype in metadata:
        total_bytes = math.prod(shape, start=1) * jnp.dtype(dtype).itemsize
        byte_chunk = blob[offset : offset + total_bytes]
        arr = byte_chunk.view(dtype).reshape(shape)
        leaves.append(arr)
        offset += total_bytes

    return jax.tree_util.tree_unflatten(treedef, leaves)


class DataTransferMode(enum.Enum):
    # Transfer data between devices as is
    UNPACKED = "unpacked"

    # Pack the leaves of all pytrees into a single byte array. This option is faster if there are many small leaves.
    PACKED = "packed"

    # Transfer the data in small chunks to avoid having to allocate a large temporary buffer on the device.
    CHUNKED = "chunked"


InputType = TypeVar("InputType")
P = ParamSpec("P")
IntermediateType = TypeVar("IntermediateType")
OutputType = TypeVar("OutputType")


class CallbackInvocationFn(Protocol[InputType, OutputType]):
    def __call__(
        self, fn: Callable[[InputType], OutputType], args: InputType
    ) -> OutputType:
        pass


class HostCallback(ABC, Generic[P, IntermediateType, OutputType]):
    def __init__(
        self,
        callback_invocation_fn: CallbackInvocationFn[IntermediateType, OutputType],
        fn: Callable[P, OutputType],
    ):
        self.__callback_invocation_fn = callback_invocation_fn
        self.__fn = fn

    def __call__(self, *args, **kwargs):
        packed_data = self._pack_data((args, kwargs))

        def fn_wrapped(packed: IntermediateType):
            try:
                args, kwargs = self._unpack_data(packed)
                return self.__fn(*args, **kwargs)
            except:
                traceback.print_exc()
                raise

        return self.__callback_invocation_fn(fn_wrapped, packed_data)

    @abstractmethod
    def _pack_data(
        self, args: tuple[tuple[Any, ...], dict[str, Any]]
    ) -> IntermediateType:
        pass

    @abstractmethod
    def _unpack_data(
        self, packed: IntermediateType
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        pass


class RawHostCallback(
    HostCallback[P, tuple[tuple[Any, ...], dict[str, Any]], OutputType],
    Generic[P, OutputType],
):
    def _pack_data(
        self, args: tuple[tuple[Any, ...], dict[str, Any]]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return args

    def _unpack_data(
        self, packed: tuple[tuple[Any, ...], dict[str, Any]]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return packed


class PackedHostCallback(
    HostCallback[P, jax.Array, OutputType],
    Generic[P, OutputType],
):
    def __init__(
        self,
        callback_invocation_fn: CallbackInvocationFn[IntermediateType, OutputType],
        fn: Callable[P, OutputType],
    ):
        super().__init__(callback_invocation_fn, fn)
        self.__packing_info = None

    def _pack_data(self, args: tuple[tuple[Any, ...], dict[str, Any]]) -> jax.Array:
        assert self.__packing_info is None, "Cannot re-use PackedHostCallback instance."
        data, self.__packing_info = pack_pytree(args)
        return data

    def _unpack_data(self, packed: jax.Array) -> tuple[tuple[Any, ...], dict[str, Any]]:
        assert (
            self.__packing_info is not None
        ), "Cannot unpack data from other PackedHostCallback."
        return unpack_pytree(packed, self.__packing_info)


@dataclass(frozen=True)
class _ChunkedCallbackSessionData:
    data: list[list[jax.Array | None]]
    tree_leaves: tuple[ShapeDtypeStruct, ...]
    tree_def: jax.tree_util.PyTreeDef

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def __restore(
        tree_def: jax.tree_util.PyTreeDef,
        tree_leaves: tuple[ShapeDtypeStruct, ...],
        data: list[list[jax.Array | None]],
    ) -> Any:
        leaves_restored = [
            jnp.concatenate(d, axis=0)[: math.prod(l.shape) * l.dtype.itemsize]
            .view(l.dtype)
            .reshape(l.shape)
            for d, l in zip(data, tree_leaves)
        ]
        return jax.tree.unflatten(tree_def, leaves_restored)

    @property
    def restored(self):
        return self.__restore(self.tree_def, self.tree_leaves, self.data)


class ChunkedHostCallback(
    HostCallback[P, jax.Array, OutputType],
    Generic[P, OutputType],
):
    def __init__(
        self,
        callback_invocation_fn: CallbackInvocationFn[IntermediateType, OutputType],
        fn: Callable[P, OutputType],
        chunk_size_bytes: int = 1024 * 1024 * 64,
    ):
        super().__init__(callback_invocation_fn, fn)
        self.__chunk_size_bytes = chunk_size_bytes
        self.__last_session_id = self.__num_session_ids - 1
        self.__send_chunk_hcb = io_callback(
            result_shape=ShapeDtypeStruct((), dtype=jnp.uint32)
        )(self.__send_chunk)
        self.__session_data: dict[int, _ChunkedCallbackSessionData] = {}

    def __create_session_factory(self, tree: Any) -> tuple[
        tuple[ShapeDtypeStruct, ...],
        jax.tree_util.PyTreeDef,
        Callable[[], jax.Array],
    ]:
        tree_leaves, tree_def = jax.tree.flatten(tree)
        tree_leaves = tuple(tree_leaves)

        def mk_session() -> jax.Array:
            session_id = (self.__last_session_id + 1) % self.__num_session_ids
            while session_id in self.__session_data:
                session_id = (session_id + 1) % self.__num_session_ids
            self.__last_session_id = session_id
            self.__session_data[session_id] = _ChunkedCallbackSessionData(
                [
                    [None] * (((b.nbytes - 1) // self.__chunk_size_bytes) + 1)
                    for b in tree_leaves
                ],
                tuple(ShapeDtypeStruct(l.shape, l.dtype) for l in tree_leaves),
                tree_def,
            )
            if len(self.__session_data) == 1000:
                logger.warning(
                    f"Reached {len(self.__session_data)} open hostcallback sessions. It is likely that sessions are "
                    f"leaking."
                )
            return jnp.array(session_id).astype(jnp.uint32)

        return tree_leaves, tree_def, mk_session

    def __send_chunk(
        self,
        session_id: jax.Array,
        object_index: jax.Array,
        chunk_index: jax.Array,
        data: jax.Array,
    ):
        sid = session_id.item()
        object_index = object_index.item()
        chunk_index = chunk_index.item()
        if sid not in self.__session_data:
            raise ValueError(f"Session with id {sid} does not exist.")
        object_data = self.__session_data[sid].data[object_index]
        if object_data[chunk_index] is not None:
            raise ValueError(
                f"Chunk {chunk_index} of object {object_index} already transferred in session {sid}."
            )
        object_data[chunk_index] = data
        return session_id

    def __transfer_data(self, tree: Any) -> jax.Array:
        tree_leaves, tree_def, mk_session = self.__create_session_factory(tree)
        session_id = io_callback(
            result_shape=ShapeDtypeStruct(shape=(), dtype=jnp.uint32)
        )(mk_session)()
        for i, l in enumerate(tree_leaves):
            data = l.reshape((-1,)).view(dtype=jnp.uint8)

            def send_chunk(j, sid):
                chunk = jax.lax.dynamic_slice_in_dim(
                    data,
                    i * self.__chunk_size_bytes,
                    min(self.__chunk_size_bytes, data.size),
                    axis=0,
                )
                # We have to pass the session ID through here to ensure that these callbacks happen before the actual
                # callback happens (I think setting ordered=True here does not work, as we need ordering across this
                # callback and the actual calling callback).
                sid = self.__send_chunk_hcb(sid, i, j, chunk)
                return sid

            session_id = jax.lax.fori_loop(
                0, (l.nbytes - 1) // self.__chunk_size_bytes + 1, send_chunk, session_id
            )
        return session_id

    @property
    def __num_session_ids(self):
        return np.iinfo(np.uint32).max

    def _pack_data(self, args: tuple[tuple[Any, ...], dict[str, Any]]) -> jax.Array:
        return self.__transfer_data(args)

    def _unpack_data(self, packed: jax.Array) -> tuple[tuple[Any, ...], dict[str, Any]]:
        session_id = packed.item()
        data = self.__session_data[session_id]
        del self.__session_data[session_id]
        return data.restored


def mk_host_callback(
    callback_invocation_fn: CallbackInvocationFn[IntermediateType, OutputType],
    fn: Callable[P, OutputType],
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> HostCallback[P, IntermediateType, OutputType]:
    if data_transfer_mode_device_to_host == DataTransferMode.UNPACKED:
        cb_type = RawHostCallback
    elif data_transfer_mode_device_to_host == DataTransferMode.PACKED:
        cb_type = PackedHostCallback
    elif data_transfer_mode_device_to_host == DataTransferMode.CHUNKED:
        cb_type = ChunkedHostCallback
    else:
        raise ValueError(
            f"Unknown device-to-host transfer mode: {data_transfer_mode_device_to_host}."
        )
    return cb_type(callback_invocation_fn, fn)


def pure_callback(
    result_shape: PyTree[jax.ShapeDtypeStruct],
    sharding=None,
    vectorized=False,
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        return mk_host_callback(
            lambda fn, args: jax.pure_callback(
                fn,
                result_shape_dtypes=result_shape,
                args=args,
                sharding=sharding,
                vectorized=vectorized,
            ),
            fn,
            data_transfer_mode_device_to_host=data_transfer_mode_device_to_host,
        )

    return decorator


def _io_callback(
    result_shape: PyTree[jax.ShapeDtypeStruct] = None,
    sharding=None,
    ordered=False,
    data_transfer_mode_host_to_device: DataTransferMode = DataTransferMode.UNPACKED,
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        def wrapped(*args, **kwargs):
            actual_fn = fn
            if data_transfer_mode_host_to_device == DataTransferMode.PACKED:
                actual_result_shape, packing_info = get_packing_info(result_shape)
                fn_old = actual_fn

                def actual_fn(*args, **kwargs):
                    return pack_pytree(fn_old(*args, **kwargs))[0]

            elif data_transfer_mode_host_to_device == DataTransferMode.UNPACKED:
                packing_info = None
                actual_result_shape = result_shape
            elif data_transfer_mode_host_to_device == DataTransferMode.CHUNKED:
                raise NotImplementedError(
                    "Chunked mode is not supported for host to device transfer."
                )
            else:
                raise ValueError(
                    f"Unknown host-to-device transfer mode: {data_transfer_mode_host_to_device}."
                )

            host_callback = mk_host_callback(
                lambda fn, args: jax.experimental.io_callback(
                    fn, actual_result_shape, args, sharding=sharding, ordered=ordered
                ),
                actual_fn,
                data_transfer_mode_device_to_host=data_transfer_mode_device_to_host,
            )
            res = host_callback(*args, **kwargs)

            if data_transfer_mode_host_to_device == DataTransferMode.PACKED:
                return unpack_pytree(res, packing_info)
            return res

        return wrapped

    return decorator


@overload
def io_callback(
    result_shape: PyTree[jax.ShapeDtypeStruct] = None,
    sharding=None,
    ordered=False,
    data_transfer_mode_host_to_device: DataTransferMode = DataTransferMode.UNPACKED,
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> Callable[[Callable], Callable]: ...


@overload
def io_callback(fn: Callable) -> Callable: ...


def io_callback(*args, **kwargs):
    if fn := _check_direct_call(args, kwargs):
        return _io_callback()(fn)
    else:
        return _io_callback(*args, **kwargs)


def _debug_callback(
    ordered: bool = False,
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> Callable[[Callable], Callable]:
    def decorator(fn: Callable) -> Callable:
        return mk_host_callback(
            lambda fn, args: jax.debug.callback(fn, args, ordered=ordered),
            fn,
            data_transfer_mode_device_to_host=data_transfer_mode_device_to_host,
        )

    return decorator


@overload
def debug_callback(
    ordered: bool = False,
    data_transfer_mode_device_to_host: DataTransferMode = DataTransferMode.UNPACKED,
) -> Callable[[Callable], Callable]: ...


@overload
def debug_callback(fn: Callable) -> Callable: ...


def debug_callback(*args, **kwargs):
    if fn := _check_direct_call(args, kwargs):
        return _debug_callback()(fn)
    else:
        return _debug_callback(*args, **kwargs)


def is_primitive_jax_type(obj) -> bool:
    return any(isinstance(obj, t) for t in (int, float, bool, jax.Array))


def is_jax_type(obj) -> bool:
    return jax.tree.all(jax.tree.map(is_primitive_jax_type, obj))


def bp(fn: Callable) -> Callable:
    _stack = inspect.stack()[1:]
    _locals = _stack[0].frame.f_locals
    l_jax_vars = {k: v for k, v in _locals.items() if is_jax_type(v)}
    l_non_jax_vars = {k: v for k, v in _locals.items() if k not in l_jax_vars}
    _globals = _stack[0].frame.f_globals
    g_jax_vars = {k: v for k, v in _globals.items() if is_jax_type(v)}
    g_non_jax_vars = {k: v for k, v in _globals.items() if k not in g_jax_vars}

    @debug_callback
    def inner(l_jax_vars, g_jax_vars):
        new_globals = {**g_jax_vars, **g_non_jax_vars}
        new_locals = {**l_jax_vars, **l_non_jax_vars, "_stack": _stack}
        exec(fn.__code__, {**new_globals, **new_locals}, new_locals)

    inner(l_jax_vars, g_jax_vars)

    return fn
