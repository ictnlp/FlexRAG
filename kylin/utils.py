import json
import os
import subprocess
from contextlib import contextmanager
from csv import reader
from enum import Enum
from functools import partial
from itertools import zip_longest
from logging import Logger
from multiprocessing import Manager
from time import perf_counter
from typing import Iterable, Iterator, Optional

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


class Register:
    def __init__(self, register_name: str = None):
        self.name = register_name
        self._items = {}
        self._shortcuts = {}
        return

    def __call__(self, *short_names: str, config_class=None):
        def registe_item(item):
            main_name = str(item).split(".")[-1][:-2]
            # check name conflict
            assert main_name not in self._items, f"Name Conflict {main_name}"
            assert main_name not in self._shortcuts, f"Name Conflict {main_name}"
            for name in short_names:
                assert name not in self._items, f"Name Conflict {name}"
                assert name not in self._shortcuts, f"Name Conflict {name}"

            # register the item
            self._items[main_name] = {
                "item": item,
                "main_name": main_name,
                "short_names": short_names,
                "config_class": config_class,
            }
            for name in short_names:
                self._shortcuts[name] = main_name
            return item

        return registe_item

    def __iter__(self):
        return self._items.__iter__()

    @property
    def names(self) -> list[str]:
        return list(self._items.keys()) + list(self._shortcuts.keys())

    @property
    def mainnames(self) -> list[str]:
        return list(self._items.keys())

    @property
    def shortnames(self) -> list[str]:
        return list(self._shortcuts.keys())

    def __getitem__(self, key: str) -> dict:
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]

    def get(self, key: str, default=None) -> dict:
        if key not in self._items:
            if key not in self._shortcuts:
                return default
            key = self._shortcuts[key]
        return self._items[key]

    def get_item(self, key: str):
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]["item"]

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self.names

    def __str__(self) -> str:
        data = {
            "name": self.name,
            "items": [
                {
                    "main_name": k,
                    "short_names": v["short_names"],
                    "config_class": str(v["config_class"]),
                }
                for k, v in self._items.items()
            ],
        }
        return json.dumps(data, indent=4)

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, register: "Register"):
        new_register = Register()
        new_register._items = {**self._items, **register._items}
        new_register._shortcuts = {**self._shortcuts, **register._shortcuts}
        return new_register


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
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if hasattr(obj, "to_list"):
            return obj.to_list()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


json.dumps = partial(json.dumps, cls=CustomEncoder)
json.dump = partial(json.dump, cls=CustomEncoder)


def read_data(
    file_paths: list[str] | str,
    data_ranges: Optional[list[list[int, int]] | list[int, int]] = None,
) -> Iterator:
    if isinstance(file_paths, str):
        file_paths = [file_paths]
        if data_ranges is not None:
            assert isinstance(data_ranges[0], int), "Invalid data ranges"
            assert isinstance(data_ranges[1], int), "Invalid data ranges"
            data_ranges = [data_ranges]
    if data_ranges is None:
        data_ranges = []

    for file_path, data_range in zip_longest(
        file_paths, data_ranges, fillvalue=[0, -1]
    ):
        start_point, end_point = data_range
        if end_point > 0:
            assert end_point > start_point, f"Invalid data range: {data_range}"
        if file_path.endswith(".jsonl"):
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i < start_point:
                        continue
                    if (end_point > 0) and (i >= end_point):
                        break
                    yield json.loads(line)
        elif file_path.endswith(".tsv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f, delimiter="\t")):
                    if i == 0:
                        title = row
                        continue
                    if i <= start_point:
                        continue
                    if (end_point > 0) and (i > end_point):
                        break
                    yield dict(zip(title, row))
        elif file_path.endswith(".csv"):
            title = []
            with open(file_path, "r") as f:
                for i, row in enumerate(reader(f)):
                    if i == 0:
                        title = row
                        continue
                    if i <= start_point:
                        continue
                    if (end_point > 0) and (i > end_point):
                        break
                    yield dict(zip(title, row))
        else:
            raise ValueError(f"Unsupported file format: {file_path}")


class _TimeMeterClass:
    def __init__(self):
        self._manager = Manager()
        self.timers = self._manager.dict()
        return

    def __call__(self, *timer_names: str):
        def time_it(func):
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = func(*args, **kwargs)
                end_time = perf_counter()
                if timer_names not in self.timers:
                    self.timers[timer_names] = self._manager.list()
                self.timers[timer_names].append(end_time - start_time)
                return result

            return wrapper

        return time_it

    @property
    def statistics(self) -> list[dict[str, float]]:
        statistics = []
        for k, v in self.timers.items():
            v = list(v)
            statistics.append(
                {
                    "name": k,
                    "calls": len(v),
                    "average call time": np.mean(v),
                    "total time": np.sum(v),
                }
            )
        return statistics

    @property
    def details(self) -> dict:
        return {k: v for k, v in self.timers.items()}


TimeMeter = _TimeMeterClass()

try:
    COMMIT_ID = (
        subprocess.check_output(
            ["git", "-C", f"{os.path.dirname(__file__)}", "rev-parse", "HEAD"]
        )
        .strip()
        .decode("utf-8")
    )
except:
    COMMIT_ID = "Unknown"
