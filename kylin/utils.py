import json
import os
from contextlib import contextmanager
from csv import reader
from enum import Enum
from functools import partial
from logging import Logger
from time import perf_counter
from typing import Iterable, Iterator

import numpy as np
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


class SimpleProgressLogger:
    def __init__(self, logger: Logger, total: int = None, interval: int = 100):
        self.total = total
        self.interval = interval
        self.logger = logger
        self.current = 0
        self.current_stage = 0
        self.desc = "Progress"
        self.start_time = perf_counter()
        return

    def update(self, step: int = 1, desc: str = None) -> None:
        if desc is not None:
            self.desc = desc
        self.current += step
        stage = self.current // self.interval
        if stage > self.current_stage:
            self.current_stage = stage
            self.log()
        return

    def log(self) -> None:
        def fmt_time(time: float) -> str:
            if time < 60:
                return f"{time:.2f}s"
            if time < 3600:
                return f"{time//60:02.0f}:{time%60:02.0f}"
            else:
                return f"{time//3600:.0f}:{(time%3600)//60:02.0f}:{time%60:02.0f}"

        if (self.total is not None) and (self.current < self.total):
            time_spend = perf_counter() - self.start_time
            time_left = time_spend * (self.total - self.current) / self.current
            speed = self.current / time_spend
            num_str = f"{self.current} / {self.total}"
            percent_str = f"({self.current/self.total:.2%})"
            time_str = f"[{fmt_time(time_spend)} / {fmt_time(time_left)}, {speed:.2f} update/s]"
            self.logger.info(f"{self.desc}: {num_str} {percent_str} {time_str}")
        else:
            time_spend = perf_counter() - self.start_time
            speed = self.current / time_spend
            num_str = f"{self.current}"
            time_str = f"[{fmt_time(time_spend)}, {speed:.2f} update/s]"
            self.logger.info(f"{self.desc}: {num_str} {time_str}")
        return


@contextmanager
def set_env_var(key, value):
    original_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if original_value is None:
            del os.environ[key]
        else:
            os.environ[key] = original_value


class StrEnum(Enum):
    def __eq__(self, other: str):
        return self.value == other

    def __str__(self):
        return self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return self.value


def Choices(choices: Iterable[str]):
    return StrEnum("Choices", {c: c for c in choices})


# Monkey Patching the JSONEncoder to handle StrEnum
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, StrEnum):
            return str(obj)
        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
        if isinstance(np.int64):
            return int(obj)
        if isinstance(np.int32):
            return int(obj)
        if isinstance(np.float64):
            return float(obj)
        if isinstance(np.float32):
            return float(obj)
        return super().default(obj)


json.dumps = partial(json.dumps, cls=CustomEncoder)
json.dump = partial(json.dump, cls=CustomEncoder)


def read_data(file_paths: list[str]) -> Iterator:
    for file_path in file_paths:
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as f:
                for line in f:
                    yield json.loads(line)
        elif file_path.endswith(".tsv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f, delimiter="\t")):
                    if i == 0:
                        title = row
                    else:
                        yield dict(zip(title, row))
        elif file_path.endswith(".csv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f)):
                    if i == 0:
                        title = row
                    else:
                        yield dict(zip(title, row))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
