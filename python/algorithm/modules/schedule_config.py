from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import chex
import flax.struct
import jax.numpy as jnp
import optax
from optax._src import numerics


class ScheduleType(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"


@flax.struct.dataclass
class NormalizedScheduleWrapperState:
    count: chex.Numeric
    limit: chex.Numeric


@dataclass(frozen=True)
class NormalizedScheduleWrapper:
    schedule_fn: Callable[[chex.Numeric], chex.Numeric]
    initial_limit: chex.Numeric

    def init(self) -> NormalizedScheduleWrapperState:
        return NormalizedScheduleWrapperState(
            jnp.zeros((), dtype=jnp.int32), self.initial_limit
        )

    def update(
        self,
        state: NormalizedScheduleWrapperState,
        **extra_args,
    ) -> NormalizedScheduleWrapperState:
        del extra_args
        new_count = numerics.safe_increment(state.count)
        return state.replace(count=new_count)

    def __call__(
        self,
        state: NormalizedScheduleWrapperState,
        **extra_args,
    ) -> chex.Numeric:
        progress = jnp.clip(state.count / (state.limit - 1), 0.0, 1.0)
        return self.schedule_fn(progress)


@dataclass
class ScheduleConfig:
    schedule_type: ScheduleType  # Which schedule to use
    value: float  # The (maximum) value of the schedule
    initial_value: float  # The initial value of the schedule
    final_value: float  # The final value of the schedule
    warmup_rel: float  # The share of the warmup steps in the total number
    exp_decay_rate: float  # The decay rate for the exponential schedule

    def make_schedule(self, total_step_count: float) -> NormalizedScheduleWrapper:
        remaining_share = 1.0 - self.warmup_rel
        if self.schedule_type == ScheduleType.CONSTANT:
            schedule = optax.constant_schedule(self.value)
        elif self.schedule_type == ScheduleType.LINEAR:
            schedule = optax.linear_schedule(
                init_value=self.value,
                end_value=self.final_value,
                transition_steps=remaining_share,
            )
        elif self.schedule_type == ScheduleType.COSINE:
            schedule = optax.cosine_decay_schedule(
                init_value=self.value,
                decay_steps=remaining_share,
                alpha=self.final_value / self.value,
            )
        elif self.schedule_type == ScheduleType.EXPONENTIAL:
            schedule = optax.exponential_decay(
                init_value=self.value,
                transition_steps=remaining_share
                / math.log(self.final_value / self.value, self.exp_decay_rate),
                decay_rate=self.exp_decay_rate,
            )
        else:
            raise NotImplementedError(f"Unknown schedule {self.type}")
        if self.warmup_rel > 0:
            schedule = optax.join_schedules(
                [
                    optax.linear_schedule(
                        init_value=self.initial_value,
                        end_value=self.value,
                        transition_steps=self.warmup_rel,
                    ),
                    schedule,
                ],
                [self.warmup_rel],  # type: ignore[arg-type]
            )
        return NormalizedScheduleWrapper(
            schedule, initial_limit=jnp.array(total_step_count)
        )
