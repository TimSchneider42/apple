from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp

from ring_buffer import RingBuffer


class RingBufferTest(unittest.TestCase):
    @staticmethod
    def _generate_random_array(
        shape: tuple[int, ...], dtype: jax.dtypes, rng: jax.Array
    ) -> jax.Array:
        if dtype == jnp.bool_:
            return jax.random.randint(rng, shape, 0, 2) == 1
        else:
            return (
                jax.random.uniform(rng, shape, dtype=jnp.float32, minval=0, maxval=1)
                * 1000
            ).astype(dtype)

    @classmethod
    def _generate_random_pytree(
        cls,
        variable_specs: dict[str, jax.ShapeDtypeStruct],
        rng: jax.Array,
        length: int,
        bs: int,
    ) -> dict[str, jax.Array]:
        rng_tree = jax.random.split(rng, len(variable_specs))
        return {
            k: cls._generate_random_array((bs, length) + s.shape, s.dtype, r)
            for (k, s), r in zip(variable_specs.items(), rng_tree)
        }

    def test(self):
        variable_specs = {
            "var_int": jax.ShapeDtypeStruct((10, 15), dtype=jnp.int32),
            "var_bool": jax.ShapeDtypeStruct((), dtype=jnp.bool_),
            "var_float": jax.ShapeDtypeStruct((4,), dtype=jnp.float32),
        }

        for i in range(10):
            with self.subTest("add", i=i):
                rng = jax.random.PRNGKey(0)
                rng, rng_data = jax.random.split(rng)
                length = 4096
                rng, rng_capacity, rng_batch_size = jax.random.split(rng, 3)
                capacity = jax.random.randint(rng_capacity, (), 1, 64).item()
                batch_size = jax.random.randint(rng_batch_size, (), 1, 8).item()
                data = self._generate_random_pytree(
                    variable_specs, rng_data, length, batch_size
                )
                buffer = RingBuffer.build(
                    variable_specs, batch_shape=(batch_size,), capacity=capacity
                )
                index = 0

                while index < length:
                    rng, rng_round, rng_shrink, rng_add = jax.random.split(rng, 4)
                    remaining = length - index
                    add_length = jax.random.randint(
                        rng_round, (), 0, min(buffer.capacity, remaining)
                    ).item()
                    if add_length == 0:
                        buffer = buffer.add_single(
                            jax.tree.map(lambda d: d[:, index], data)
                        )
                        index += 1
                    else:
                        buffer = buffer.add(
                            jax.tree.map(
                                lambda d: d[:, index : index + add_length], data
                            )
                        )
                        index += add_length
                    expected_content = jax.tree.map(
                        lambda d: d[:, max(0, index - buffer.capacity) : index], data
                    )
                    actual_content = buffer.valid_data
                    self.assertTrue(
                        jax.tree.all(
                            jax.tree.map(
                                lambda a, b: jnp.all(a == b),
                                expected_content,
                                actual_content,
                            )
                        )
                    )

                    new_capacity = jax.random.randint(
                        rng_shrink, (), 0, buffer.capacity
                    ).item()
                    buffer_shrunk = buffer.resize(new_capacity)

                    self.assertEqual(buffer_shrunk.capacity, new_capacity)

                    self.assertEqual(
                        buffer_shrunk.length, min(buffer.length, buffer_shrunk.capacity)
                    )

                    expected_content = jax.tree.map(
                        lambda d: d[:, max(0, index - buffer_shrunk.capacity) : index],
                        data,
                    )

                    self.assertTrue(
                        jax.tree.all(
                            jax.tree.map(
                                lambda a, b: jnp.all(a == b),
                                expected_content,
                                buffer_shrunk.valid_data,
                            )
                        )
                    )

                    new_capacity = jax.random.randint(
                        rng_shrink, (), buffer.capacity, 5 * buffer.capacity
                    ).item()
                    buffer_grown = buffer.resize(new_capacity)

                    self.assertEqual(buffer_grown.capacity, new_capacity)

                    self.assertEqual(buffer_grown.length, buffer.length)

                    expected_content = jax.tree.map(
                        lambda d: d[:, max(0, index - buffer.capacity) : index], data
                    )

                    self.assertTrue(
                        jax.tree.all(
                            jax.tree.map(
                                lambda a, b: jnp.all(a == b),
                                expected_content,
                                buffer_grown.valid_data,
                            )
                        )
                    )
