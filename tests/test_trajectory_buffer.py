from __future__ import annotations

import unittest
from itertools import chain

import jax
import jax.numpy as jnp
import numpy as np

from trajectory_buffer import TrajectoryBuffer


class TrajectoryBufferTest(unittest.TestCase):
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
    def _generate_trajectory(
        cls,
        variable_specs: dict[str, jax.ShapeDtypeStruct],
        rng: jax.Array,
        traj_idx: int,
        max_len: int = 50,
    ) -> dict[str, jax.Array]:
        rng, rng_tmp = jax.random.split(rng)
        rng_tree = jax.random.split(rng, len(variable_specs))
        trajectory_length = jax.random.randint(rng_tmp, (), 2, max_len).item()
        output = {
            k: cls._generate_random_array((trajectory_length,) + s.shape, s.dtype, r)
            for (k, s), r in zip(variable_specs.items(), rng_tree)
            if k != "traj_idx"
        }
        output["done"] = jnp.zeros((trajectory_length,), dtype=jnp.bool_)
        output["done"] = output["done"].at[-1].set(True)
        output["traj_idx"] = jnp.full((trajectory_length,), traj_idx, dtype=jnp.int32)
        output["traj_step"] = jnp.arange(trajectory_length, dtype=jnp.int32)
        return output

    def test(self):
        batch_size = 4
        traj_count = 10
        traj_max_len = 5
        rng = jax.random.PRNGKey(0)
        rng, rng_tmp = jax.random.split(rng)
        rng_traj = jax.random.split(rng_tmp, traj_count * batch_size)

        int_shape = (10, 15)
        float_shape = (4,)
        bool_shape = ()

        variable_specs = {
            "var_int": jax.ShapeDtypeStruct(int_shape, dtype=jnp.int32),
            "var_bool": jax.ShapeDtypeStruct(bool_shape, dtype=jnp.bool_),
            "var_float": jax.ShapeDtypeStruct(float_shape, dtype=jnp.float32),
            "traj_idx": jax.ShapeDtypeStruct((), dtype=jnp.int32),
            "traj_step": jax.ShapeDtypeStruct((), dtype=jnp.int32),
        }
        trajectories = [
            [
                self._generate_trajectory(
                    variable_specs, r, i + traj_count * b, max_len=traj_max_len
                )
                for i, r in enumerate(rng_traj[b * traj_count : (b + 1) * traj_count])
            ]
            for b in range(batch_size)
        ]
        traj_min_length = min(
            sum(t["done"].shape[0] for t in traj) for traj in trajectories
        )
        trajectories_by_idx = list(chain(*trajectories))

        data = {
            k: jnp.stack(
                [
                    jnp.concatenate([t[k] for t in traj])[:traj_min_length]
                    for traj in trajectories
                ],
                axis=0,
            )
            for k in chain(variable_specs, ["done"])
        }

        tb = TrajectoryBuffer.build(
            variable_specs, capacity=int(traj_min_length * 0.8), stream_count=batch_size
        )

        trajectories_added = jnp.zeros((batch_size,), dtype=jnp.int32)
        trajectories_started = jnp.ones((batch_size,), dtype=jnp.int32)
        last_done = jnp.ones((batch_size,), dtype=np.int32)
        for i in range(traj_min_length):
            tb = tb.add_step(
                data["done"][:, i],
                {k: v[:, i] for k, v in data.items() if k != "done"},
            )
            trajectories_added += data["done"][:, i].astype(jnp.int32)
            if i > 0:
                trajectories_started += data["done"][:, i - 1].astype(jnp.int32)
            last_done = last_done.at[data["done"][:, i]].set(0)
            reverse_space_requirements = [
                np.cumsum(
                    np.flip(
                        [
                            t["done"].shape[0]
                            for t in trajectories[j][: trajectories_added[j]]
                        ],
                        axis=0,
                    )
                )
                for j in range(batch_size)
            ]
            fitting_trajectories = [
                list(
                    reversed(
                        [
                            traj_added - j - 1
                            for j, ts in enumerate(req)
                            if ts + ld <= tb.capacity_steps
                        ]
                    )
                )
                for traj_added, req, ld in zip(
                    np.array(trajectories_added),
                    reverse_space_requirements,
                    last_done,
                )
            ]
            present_trajectories = [
                list(
                    reversed(
                        [
                            traj_added - j - 1
                            for j, ts in enumerate(
                                np.concatenate([[1], np.array(req[:-1]) + 1])[
                                    : len(req)
                                ]
                            )
                            if ts + ld <= tb.capacity_steps
                        ]
                    )
                )
                + ([traj_started - 1] if ld != 0 else [])
                for traj_started, traj_added, req, ld in zip(
                    np.array(trajectories_started),
                    np.array(trajectories_added),
                    reverse_space_requirements,
                    last_done,
                )
            ]

            with self.subTest("trajectory count", i=i):
                self.assertEqual(
                    sum(map(len, fitting_trajectories)), tb.complete_trajectory_count
                )

            fitting_trajectories_flat = set(
                traj[t]["traj_idx"][0].item()
                for traj, ts in zip(trajectories, fitting_trajectories)
                for t in ts
            )
            with self.subTest("trajectory integrity", i=i):
                found_trajectories = set()
                for j in range(tb.complete_trajectory_count):
                    traj = tb.get_trajectory(j).variables
                    traj_idx = traj["traj_idx"][0].item()
                    found_trajectories.add(traj_idx)
                    self.assertTrue(
                        all(
                            k in traj.keys()
                            for k in trajectories_by_idx[traj_idx].keys()
                            if k != "done"
                        )
                    )
                    for k, v in traj.items():
                        if k in trajectories_by_idx[traj_idx]:
                            self.assertTrue(
                                jnp.all(v == trajectories_by_idx[traj_idx][k]).item()
                            )
                self.assertEqual(found_trajectories, fitting_trajectories_flat)

            present_trajectories_flat = set(
                traj[t]["traj_idx"][0].item()
                for traj, ts in zip(trajectories, present_trajectories)
                for t in ts
            )
            with self.subTest("partial trajectories", i=i):
                found_partial_trajectories = set(
                    np.array(tb.valid_data.variables["traj_idx"].reshape(-1)).tolist()
                )
                self.assertEqual(found_partial_trajectories, present_trajectories_flat)

            sampled_trajectories = set()
            with self.subTest("sample batch", i=i):
                # In the worst case, the replay buffer is full, and a large trajectory at the beginning has just been
                # started to be overwritten. In this case, the maximum sequence length is the buffer capacity minus the
                # length of the trajectory that is currently being overwritten.
                max_seq_length = min(tb.capacity_steps - traj_max_len - 1, i + 1)
                total_samples_drawn = 0
                if max_seq_length > 0:
                    while total_samples_drawn < 5 * len(present_trajectories_flat):
                        (
                            rng,
                            rng_sequence_length,
                            rng_batch_size,
                            rng_sample,
                        ) = jax.random.split(rng, 4)
                        sequence_length = jax.random.randint(
                            rng_sequence_length, (), 1, max_seq_length
                        ).item()
                        bs = jax.random.randint(
                            rng_batch_size, (), 1, batch_size
                        ).item()
                        total_samples_drawn += bs
                        batch = tb.sample_batch(rng_sample, bs, sequence_length)
                        self.assertEqual(
                            batch.data.variables["traj_idx"].shape,
                            (bs, sequence_length),
                        )
                        self.assertTrue(batch.data_valid.all())
                        self.assertTrue(
                            (
                                batch.data.start
                                == (batch.data.variables["traj_step"] == 0)
                            )
                            .all()
                            .item()
                        )
                        self.assertTrue(
                            (batch.data.step_no == batch.data.variables["traj_step"])
                            .all()
                            .item()
                        )
                        for j in range(bs):
                            traj_idx = jnp.unique(batch.data.variables["traj_idx"][j])
                            sampled_trajectories.update(traj_idx.tolist())
                            for idx in traj_idx:
                                traj_mask = batch.data.variables["traj_idx"][j] == idx
                                traj_steps = batch.data.variables["traj_step"][j][
                                    traj_mask
                                ]
                                orig_traj = trajectories_by_idx[idx]
                                if len(traj_steps) == len(orig_traj["traj_step"]):
                                    self.assertTrue(
                                        batch.data.end[j][traj_mask][-1].item()
                                    )
                                for k in orig_traj.keys():
                                    if k != "done":
                                        self.assertTrue(
                                            jnp.all(
                                                batch.data.variables[k][j][traj_mask]
                                                == orig_traj[k][traj_steps]
                                            ).item()
                                        )
                                        l = traj_mask.sum()
                                        # self.assertEqual(
                                        #     jnp.all(
                                        #         batch.trajectory_complete[j][traj_mask]
                                        #     ),
                                        #     l == len(orig_traj[k])
                                        #     and idx.item() in fitting_trajectories_flat,
                                        # )
                self.assertLessEqual(sampled_trajectories, present_trajectories_flat)
                self.assertLessEqual(
                    len(present_trajectories_flat - sampled_trajectories)
                    - jnp.sum(last_done == 1).item(),
                    0.15 * len(present_trajectories_flat),
                )
            last_done += 1
