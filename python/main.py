#!/usr/bin/env python3
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import VectorizeMode
from gymnasium.envs.registration import parse_env_id
from hydra.core.config_store import ConfigStore
from jaxlib.xla_client import XlaRuntimeError
from omegaconf import OmegaConf

from algorithm import BaseAlgorithmConfig, PPOConfig, HAMConfig, SACConfig
from algorithm.wrappers import DiscreteActionFlattenWrapper
from data_logger import DataLogger
from hydra_resolvers import register_hydra_resolvers
from envs import register_envs

logger = logging.getLogger(__name__)

register_envs()
register_hydra_resolvers()


def mk_seed(rng: np.random.Generator):
    return int(rng.integers(0, np.iinfo(np.int32).max))


cs = ConfigStore.instance()
cs.store(group="algorithm", name="ppo_schema", node=PPOConfig)
cs.store(group="algorithm", name="ham_schema", node=HAMConfig)
cs.store(group="algorithm", name="sac_schema", node=SACConfig)


@dataclass
class Config:
    algorithm: BaseAlgorithmConfig

    # Base output directory, in which an unqiue directory will be placed
    base_output_dir: str

    # Seed of the experiment
    seed: int | None

    # If set, this experiment will be tracked with Weights and Biases
    wandb: bool

    # The wandb's project name
    wandb_project_name: str

    # The entity (team) of wandb's project
    wandb_entity: str | None

    # Group of this experiment
    wandb_group: str | None

    # Subgroup of this experiment
    wandb_subgroup: str | None

    # Supergroup of this experiment
    wandb_supergroup: str | None

    # The ID of the environment
    env_id: str

    # Disables JAX JIT compilation
    no_jit: bool

    # Number of environments to run in parallel
    num_envs: int

    # Use a separate environment for evaluation
    use_eval_env: bool

    # ID of the evaluation environment. Defaults to the training environment
    eval_env_id: str | None

    # Enable persistent JIT cache
    enable_persistent_jit_cache: bool

    # If set, only evaluation will be performed
    evaluation_mode: bool

    # Number of steps per checkpoint in evaluation mode
    evaluation_mode_num_steps_per_cp: int

    # Additional keyword arguments for the training environment
    env_kwargs: dict

    # Additional keyword arguments for the evaluation environment
    eval_env_kwargs: dict | None


def get_env_specific_kwargs(env_id: str) -> dict[str, Any]:
    kwargs = {"render_mode": "rgb_array"}
    ns, name, version = parse_env_id(env_id)
    name_split = name.split(":")
    if len(name_split) == 2:
        module = name_split[0]
    else:
        module = None
    if module == "mikasa_robo_suite":
        kwargs.update(
            dict(wrappers=[FixMikasaWrapper], obs_mode="sensor_data"),
            vectorization_mode=VectorizeMode.SYNC,
        )
    elif module in ["memory_gym", "popgym"]:
        wrappers = [DiscreteActionFlattenWrapper]
        if module == "popgym" and "MineSweeper" in name:
            wrappers.insert(0, FixMinesweeperWrapper)
        if module == "popgym":
            # Not supported
            del kwargs["render_mode"]
            wrappers.append(TextToImageWrapper)
        kwargs.update(
            dict(
                wrappers=wrappers,
                vectorization_mode=VectorizeMode.SYNC,
            )
        )
    return kwargs


@hydra.main(version_base="1.3", config_path="../config", config_name="default")
def main(raw_config: Config):
    # Convert to structured config for validation
    structured_cfg = OmegaConf.structured(Config)
    config_dict = OmegaConf.merge(structured_cfg, raw_config)
    config = OmegaConf.to_object(config_dict)

    if config.enable_persistent_jit_cache:
        jax_cache_path = Path.home() / ".cache" / "jax"
        jax_cache_path.mkdir(exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", str(jax_cache_path))
        if "jax_persistent_cache_enable_xla_caches" in jax.config.values:
            jax.config.update("jax_persistent_cache_enable_xla_caches", "all")

        try:
            jnp.empty(())
        except XlaRuntimeError:
            logger.warning(
                "Disabling persistent JIT cache due to XLA error. See https://github.com/jax-ml/jax/issues/28247 for "
                "details."
            )
            jax.config.update("jax_compilation_cache_dir", None)

    if config.seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    else:
        seed = config.seed

    logger.info(f"Using seed {seed}.")
    np_rng = np.random.default_rng(seed)

    run_directory = Path(hydra.core.hydra_config.HydraConfig.get().run.dir).resolve()
    run_name = run_directory.name
    logger.info(f"Logging into {run_directory}.")

    dl = DataLogger(
        run_directory,
        OmegaConf.to_container(config_dict),
        use_tensorboard=True,
        use_wandb=config.wandb,
        wandb_project_name=config.wandb_project_name,
        wandb_entity=config.wandb_entity,
        wandb_group=config.wandb_group,
        wandb_run_name=run_name,
    )

    random.seed(mk_seed(np_rng))
    np.random.seed(mk_seed(np_rng))
    rng = jax.random.PRNGKey(mk_seed(np_rng))

    env = gym.make_vec(
        config.env_id,
        num_envs=config.num_envs,
        **get_env_specific_kwargs(config.env_id),
        **config.env_kwargs,
    )
    env.action_space.seed(mk_seed(np_rng))
    env.observation_space.seed(mk_seed(np_rng))

    if config.use_eval_env:
        eval_env = gym.make_vec(
            config.env_id if config.eval_env_id is None else config.eval_env_id,
            num_envs=config.num_envs,
            **get_env_specific_kwargs(config.env_id),
            **(
                config.eval_env_kwargs
                if config.eval_env_kwargs is not None
                else config.env_kwargs
            ),
        )
        eval_env.action_space.seed(mk_seed(np_rng))
        eval_env.observation_space.seed(mk_seed(np_rng))
    else:
        eval_env = None

    algorithm = config.algorithm.make_algorithm()
    with jax.disable_jit(config.no_jit):
        try:
            if config.evaluation_mode:
                algorithm.evaluate(
                    rng,
                    env,
                    dl,
                    config.evaluation_mode_num_steps_per_cp,
                    eval_env=eval_env,
                )
            else:
                algorithm.run(rng, env, dl, eval_env=eval_env)
        finally:
            env.close()
            if eval_env is not None:
                eval_env.close()


if __name__ == "__main__":
    main()
