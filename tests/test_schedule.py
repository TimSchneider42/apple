import math
import random

import jax.numpy as jnp
import pytest

from algorithm.modules import (
    ScheduleConfig,
    ScheduleType,
)


# -------------------------
# Helpers
# -------------------------


def advance_to_count(schedule, state, target_count: int):
    """Increment the wrapper's state.count until it reaches target_count."""

    def to_int(x):
        return int(jnp.asarray(x).item())

    c = to_int(state.count)
    while c < target_count:
        state = schedule.update(state)
        c = to_int(state.count)
    return state


# -------------------------
# Randomized config generator
# -------------------------


def rand_config(seed=None, warmup_override=None):
    rng = random.Random(seed)
    value = 10 ** rng.uniform(-3, 1)  # ~[1e-3, 10]
    final_value = rng.uniform(0.0, 1.0) * value
    initial_value = rng.uniform(0.0, 1.0) * value
    warmup_rel = rng.uniform(0.0, 0.95) if warmup_override is None else warmup_override
    exp_decay_rate = rng.uniform(0.5, 0.99)
    total_steps = rng.choice([17, 33, 64, 101, 257])

    return (
        ScheduleConfig(
            schedule_type=None,  # filled later
            value=value,
            initial_value=initial_value,
            final_value=final_value,
            warmup_rel=warmup_rel,
            exp_decay_rate=exp_decay_rate,
        ),
        total_steps,
    )


# -------------------------
# Tests with randomized configs
# -------------------------


@pytest.mark.parametrize(
    "stype",
    [
        ScheduleType.CONSTANT,
        ScheduleType.LINEAR,
        ScheduleType.COSINE,
        ScheduleType.EXPONENTIAL,
    ],
)
@pytest.mark.parametrize("seed", range(10))  # 10 randomized runs per schedule type
def test_initial_and_after_warmup(stype, seed):
    cfg, total_steps = rand_config(seed=seed)
    cfg = cfg.__class__(**{**cfg.__dict__, "schedule_type": stype})
    schedule = cfg.make_schedule(total_step_count=total_steps)
    state = schedule.init()

    # At count 0
    val0 = float(schedule(state))
    if cfg.warmup_rel > 0:
        assert val0 == pytest.approx(cfg.initial_value, rel=0, abs=1e-6)

        # At warmup boundary -> must equal value
        warmup_count = math.ceil(cfg.warmup_rel * (total_steps - 1))
        state = advance_to_count(schedule, state, warmup_count)
        v_warmup = float(schedule(state))
        assert v_warmup == pytest.approx(cfg.value, rel=0, abs=1e-2)
    else:
        # If warmup=0, initial_value is ignored -> we start at value
        assert val0 == pytest.approx(cfg.value, rel=0, abs=1e-6)


@pytest.mark.parametrize(
    "stype", [ScheduleType.LINEAR, ScheduleType.COSINE, ScheduleType.EXPONENTIAL]
)
@pytest.mark.parametrize("seed", range(10))
def test_final_value_for_non_constant(stype, seed):
    cfg, total_steps = rand_config(seed=seed)
    cfg = cfg.__class__(**{**cfg.__dict__, "schedule_type": stype})
    schedule = cfg.make_schedule(total_step_count=total_steps)
    state = schedule.init()

    # Move to last step
    last_count = total_steps - 1
    state = advance_to_count(schedule, state, last_count)
    v_end = float(schedule(state))
    assert v_end == pytest.approx(cfg.final_value, rel=0, abs=1e-5)


# -------------------------
# Explicit warmup_rel = 0 test (all else random)
# -------------------------


@pytest.mark.parametrize(
    "stype",
    [
        ScheduleType.CONSTANT,
        ScheduleType.LINEAR,
        ScheduleType.COSINE,
        ScheduleType.EXPONENTIAL,
    ],
)
@pytest.mark.parametrize("seed", range(5))
def test_warmup_zero_random_params(stype, seed):
    cfg, total_steps = rand_config(seed=seed, warmup_override=0.0)
    cfg = cfg.__class__(**{**cfg.__dict__, "schedule_type": stype})
    schedule = cfg.make_schedule(total_step_count=total_steps)
    state = schedule.init()

    # At step 0 we must immediately start at `value`
    v0 = float(schedule(state))
    assert v0 == pytest.approx(cfg.value, rel=0, abs=1e-6)

    # Non-constant schedules still must reach final_value at the end
    if stype != ScheduleType.CONSTANT:
        last_count = total_steps - 1
        state = advance_to_count(schedule, state, last_count)
        v_end = float(schedule(state))
        assert v_end == pytest.approx(cfg.final_value, rel=0, abs=1e-5)
