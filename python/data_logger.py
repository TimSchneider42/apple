from __future__ import annotations

import functools
import inspect
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING, SupportsFloat, Callable

import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


class DelayedWandBCallFuture:
    def __init__(self, fn: Callable):
        self.__fn = fn
        self.__result = None
        self.__result_available = False
        self.__callbacks = []

    def _call(self, *args, **kwargs):
        assert not self.__result_available
        self.__result_available = True
        success = False
        try:
            self.__result = self.__fn(*args, **kwargs)
            success = True
        except Exception as ex:
            self.__result = ex
        if success:
            for callback in self.__callbacks:
                callback(self.__result)

    def then(self, callback: Callable):
        if self.__result_available:
            callback(self.result)
        else:
            self.__callbacks.append(callback)
        return self

    @property
    def result(self):
        if not self.__result_available:
            raise RuntimeError("Result is not available yet")
        if isinstance(self.__result, Exception):
            raise self.__result
        return self.__result


class DelayedWandB:
    # This class delays init and all subsequent calls until the first write command, in order to set the resume from
    # parameter.
    def __init__(self, lazy: bool = False):
        import wandb

        self.__wandb = wandb
        self.__futures = []
        self.__is_initialized = False
        self.__lazy = lazy

    def __getattr__(self, name):
        attr = getattr(self.__wandb, name)
        if inspect.isfunction(attr) or inspect.ismethod(attr):

            def wrapper(*args, **kwargs):
                if not self.__is_initialized and name == "log":
                    kwargs_full = {
                        **dict(
                            zip(getattr(self.__wandb, name).__code__.co_varnames, args)
                        ),
                        **kwargs,
                    }
                    step = kwargs_full.get("step", 0)
                    for future in self.__futures:
                        future._call(step)
                    self.__is_initialized = True
                if self.__is_initialized or not self.__lazy:
                    future = DelayedWandBCallFuture(getattr(self.__wandb, name))
                    future._call(*args, **kwargs)
                    return future
                else:
                    if name == "init":

                        def call(initial_step):
                            return getattr(self.__wandb, name)(
                                *args, **kwargs, resume_from=initial_step
                            )

                    else:

                        def call(initial_step):
                            return getattr(self.__wandb, name)(*args, **kwargs)

                    self.__futures.append(DelayedWandBCallFuture(call))
                    return self.__futures[-1]

            return wrapper
        else:
            return attr


class Media(ABC):
    @abstractmethod
    def to_wandb(self, wandb: DelayedWandB):
        pass

    @abstractmethod
    def log_tensorboard(self, writer: "SummaryWriter", tag: str, step: int):
        pass


@dataclass(frozen=True)
class Image(Media):
    data: np.ndarray

    def log_tensorboard(self, writer: "SummaryWriter", tag: str, step: int):
        writer.add_image(tag, np.transpose(self.data, (2, 0, 1)), step)

    def to_wandb(self, wandb: DelayedWandB):
        return wandb.Image(self.data)


@dataclass(frozen=True)
class Video(Media):
    frames: np.ndarray
    fps: int

    def to_wandb(self, wandb: DelayedWandB):
        frames = self.data_transformed
        if frames.dtype == np.floating:
            frames = frames * 255
        return wandb.Video(frames.astype(np.uint8), fps=self.fps, format="mp4")

    def log_tensorboard(self, writer: "SummaryWriter", tag: str, step: int):
        frames = self.data_transformed
        # Future self: remove the division by 1000, once the bug in moviepy is fixed
        # https://github.com/Zulko/moviepy/issues/2151
        writer.add_video(
            tag, frames.reshape((-1, *frames.shape[-4:])), step, self.fps / 1000
        )

    @functools.cached_property
    def data_transformed(self):
        if len(self.frames.shape) == 4:
            return np.transpose(self.frames, (0, 3, 1, 2))
        elif len(self.frames.shape) == 5:
            return np.transpose(self.frames, (0, 1, 4, 2, 3))
        else:
            raise ValueError("Invalid number of dimensions for video frames")


Loggable = SupportsFloat | Media | np.ndarray


class BaseDataLogger(ABC):
    def __init__(self, default_initial_step_value: int = 0):
        self.__current_step = None
        self.__default_initial_step_value = default_initial_step_value

    def write(self, data: dict[str, Loggable], step: int | None = None):
        if step is None:
            step = (
                self.__current_step
                if self.__current_step is not None
                else self.__default_initial_step_value
            )
        if self.__current_step is not None and step < self.__current_step:
            raise ValueError("Log step must be monotonically increasing")
        self._write(data, step)
        self.__current_step = step

    @abstractmethod
    def _write(self, data: dict[str, Loggable], step: int):
        pass

    @property
    def current_step(self) -> int | None:
        return self.__current_step

    @property
    def default_initial_step_value(self) -> int:
        return self.__default_initial_step_value


class DataLogger(BaseDataLogger):
    def __init__(
        self,
        run_directory: Path,
        hyperparameters: dict[str, Any],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project_name: str | None = None,
        wandb_entity: str | None = None,
        wandb_group: str | None = None,
        wandb_run_name: str | None = None,
    ):
        super().__init__()
        self.__run_directory = run_directory
        self.__hyperparameters = hyperparameters
        self.__use_tensorboard = use_tensorboard
        self.__use_wandb = use_wandb
        self.__wandb_project_name = wandb_project_name
        self.__wandb_entity = wandb_entity
        self.__wandb_group = wandb_group
        self.__wandb_run_name = wandb_run_name
        self.__tensorboard_writer: SummaryWriter | None = None
        self.__wandb = None
        self.__used = False
        self.__open = False

    def _write(self, data: dict[str, Loggable], step: int):
        if self.__tensorboard_writer is not None:
            for tag, value in data.items():
                if isinstance(value, Media):
                    value.log_tensorboard(self.__tensorboard_writer, tag, step)
                elif isinstance(value, np.ndarray):
                    self.__tensorboard_writer.add_tensor(
                        tag, torch.from_numpy(value), step
                    )
                else:
                    self.__tensorboard_writer.add_scalar(tag, value, step)
        if self.__wandb is not None:
            self.__wandb.log(
                {
                    key: (
                        value
                        if not isinstance(value, Media)
                        else value.to_wandb(self.__wandb)
                    )
                    for key, value in data.items()
                },
                step=step,
            )

    def open(self):
        assert not self.__used
        self.__used = True

        if not self.__run_directory.exists():
            self.__run_directory.mkdir()

        logs_present_file = self.__run_directory / ".data_logger"
        resume_run = logs_present_file.exists()
        logs_present_file.touch()

        augmented_hyperparameters = self.__hyperparameters.copy()

        if "SLURM_JOB_ID" in os.environ:
            augmented_hyperparameters["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

        if self.__use_wandb:
            self.__wandb = DelayedWandB(lazy=resume_run)

            wandb_id_file = self.__run_directory / ".wandb_id"
            if wandb_id_file.exists():
                with wandb_id_file.open("r") as f:
                    wandb_id = f.read()
            else:
                wandb_id = None

            def write_run_id(run):
                with wandb_id_file.open("w") as f:
                    f.write(run.id)

            self.__wandb.init(
                project=self.__wandb_project_name,
                entity=self.__wandb_entity,
                group=self.__wandb_group,
                config=augmented_hyperparameters,
                name=self.__wandb_run_name,
                id=wandb_id,
                save_code=True,
                resume="allow",
                dir=str(self.__run_directory),
                settings=self.__wandb.Settings(code_dir=Path(__file__).parent),
            ).then(write_run_id)
        else:
            self.__wandb = None

        if self.__use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.__tensorboard_writer = SummaryWriter(str(self.__run_directory))
            self.__tensorboard_writer.add_text(
                "hyperparameters", yaml.dump(augmented_hyperparameters)
            )
        else:
            self.__tensorboard_writer = None

        with (self.__run_directory / "config.json").open("w") as f:
            json.dump(self.__hyperparameters, f)

        self.__open = True

    def close(self):
        assert self.__open
        if self.__tensorboard_writer is not None:
            self.__tensorboard_writer.close()
        if self.__wandb is not None:
            self.__wandb.finish()

    def define_metric(self, name: str, step_metric: str | None = None):
        if self.__wandb is not None:
            self.__wandb.define_metric(name, step_metric=step_metric)

    def custom_axis(
        self, axis_name: str, display_axis_name: str | None = None
    ) -> "CustomAxisDataLogger":
        return CustomAxisDataLogger(self, axis_name, display_axis_name)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def run_directory(self) -> Path:
        return self.__run_directory


class CustomAxisDataLogger(BaseDataLogger):
    def __init__(
        self,
        logger: DataLogger,
        axis_name: str,
        display_axis_name: str | None = None,
    ):
        super().__init__()
        self.__logger = logger
        self.__axis_name = axis_name
        self.__registered_metrics = set()
        self.__logger.define_metric(self.full_axis_name)
        self.__last_logged_inner_step = None
        self.__display_axis_name = (
            axis_name if display_axis_name is None else display_axis_name
        )

    def _write(self, data: dict[str, Loggable], step: int):
        unregistered_metrics = set(data.keys()) - self.__registered_metrics
        for metric in unregistered_metrics:
            self.__logger.define_metric(metric, step_metric=self.full_display_axis_name)
            self.__registered_metrics.add(metric)
        axis_data = {}
        if self.current_step is None or self.current_step == step:
            inner_step = self.__logger.current_step
            if inner_step is None:
                inner_step = self.__logger.default_initial_step_value
        else:
            inner_step = self.__logger.current_step + 1
        if self.__last_logged_inner_step != inner_step:
            axis_data[self.full_axis_name] = step
        if (
            self.current_step is not None
            and self.current_step != step
            and self.__last_logged_inner_step != inner_step - 1
        ):
            # Log the previous axis value again, as it is valid until before the current step
            self.__logger.write(
                {self.full_axis_name: self.current_step}, step=inner_step - 1
            )
        self.__logger.write({**axis_data, **data}, step=inner_step)
        self.__last_logged_inner_step = inner_step

    @property
    def full_axis_name(self):
        return f"axes/{self.__axis_name}"

    @property
    def full_display_axis_name(self):
        return f"axes/{self.__display_axis_name}"
