from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

from util import make_unique_dir


# Using a cache dir here makes sure that the same directory is not created twice by the same process and the function
# always returns the same directory for the same input.
@lru_cache(maxsize=None)
def hydra_unique_dir(base_path: str, suffix: str = ""):
    # This function finds and creates a unique directory. Creating it here is necessary to ensure that no other process
    # tries to reserve the same directory in the meantime.
    return str(make_unique_dir(Path(base_path), suffix=suffix))


def register_hydra_resolvers():
    OmegaConf.register_new_resolver("unique_dir", hydra_unique_dir)
    OmegaConf.register_new_resolver(
        "project_root", lambda: Path(__file__).resolve().parents[1]
    )
