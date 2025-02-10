import importlib
import json
import logging
import os
import sys
import threading
from contextlib import contextmanager
from dataclasses import field, make_dataclass
from enum import Enum
from functools import partial
from multiprocessing import Manager
from time import perf_counter
from typing import Coroutine, Generic, Iterable, Optional, TypeVar

import colorama
import numpy as np
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf


colorama.init(autoreset=True)


class SimpleProgressLogger:
    def __init__(self, logger: logging.Logger, total: int = None, interval: int = 100):
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

    def __repr__(self) -> str:
        return f"ProgressLogger({self.current}/{self.total})"


RegistedType = TypeVar("RegistedType")


class Register(Generic[RegistedType]):
    def __init__(self, register_name: str = None):
        self.name = register_name
        self._items = {}
        self._shortcuts = {}
        return

    def __call__(self, *short_names: str, config_class=None):
        """Register an item to the register.

        :param short_names: The short names of the item.
        :type short_names: str
        :param config_class: The config class of the item, defaults to None.
        :type config_class: dataclass
        :return: The item.
        :rtype: Any
        """

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
        """Get the names of the registered items."""
        return list(self._items.keys()) + list(self._shortcuts.keys())

    @property
    def mainnames(self) -> list[str]:
        """Get the main names of the registered items."""
        return list(self._items.keys())

    @property
    def shortnames(self) -> list[str]:
        """Get the short names of the registered items."""
        return list(self._shortcuts.keys())

    def __getitem__(self, key: str) -> dict:
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]

    def get(self, key: str, default=None) -> dict:
        """Get the item dict by name.

        :param key: The name of the item.
        :type key: str
        :param default: The default value to return, defaults to None.
        :type default: Any
        :return: The item dict containing the item, main_name, short_names, and config_class.
        :rtype: dict
        """
        if key not in self._items:
            if key not in self._shortcuts:
                return default
            key = self._shortcuts[key]
        return self._items[key]

    def get_item(self, key: str):
        """Get the item by name.

        :param key: The name of the item.
        :type key: str
        :return: The item.
        :rtype: Any
        """
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]["item"]

    def make_config(
        self,
        allow_multiple: bool = False,
        default: Optional[str] = MISSING,
        config_name: str = None,
    ):
        """Make a config class for the registered items.

        :param allow_multiple: Whether to allow multiple items to be selected, defaults to False.
        :param default: The default item to select, defaults to MISSING(???).
        :param config_name: The name of the config class, defaults to None.
        :type allow_multiple: bool, optional
        :type default: Optional[str], optional
        :type config_name: str, optional
        :return: The config class.
        :rtype: dataclass
        """
        choice_name = f"{self.name}_type"
        config_name = f"{self.name}_config" if config_name is None else config_name
        if allow_multiple:
            config_fields = [
                (
                    choice_name,
                    list[Choices(self.names)],
                    field(default_factory=list),
                )
            ]
        else:
            config_fields = [
                (
                    choice_name,
                    Optional[Choices(self.names)],
                    field(default=default),
                )
            ]
        config_fields += [
            (
                f"{self[name]['short_names'][0]}_config",
                self[name]["config_class"],
                field(default_factory=self._items[name]["config_class"]),
            )
            for name in self.mainnames
            if self[name]["config_class"] is not None
        ]
        generated_config = make_dataclass(config_name, config_fields)

        # set docstring
        docstring = (
            f"Configuration class for {self.name} "
            f"(name: {config_name}, default: {default}).\n\n"
        )
        docstring += f":param {choice_name}: The {self.name} type to use.\n"
        if allow_multiple:
            docstring += f":type {choice_name}: list[str]\n"
        else:
            docstring += f":type {choice_name}: str\n"
        for name in self.mainnames:
            if self[name]["config_class"] is not None:
                docstring += f":param {self[name]['short_names'][0]}_config: The config for {name}.\n"
                docstring += f":type {self[name]['short_names'][0]}_config: {self[name]['config_class'].__name__}\n"
        generated_config.__doc__ = docstring
        return generated_config

    def load(self, config: DictConfig, **kwargs) -> RegistedType | list[RegistedType]:
        """Load the item(s) from the generated config.

        :param config: The config generated by `make_config` method.
        :type config: DictConfig
        :return: The loaded item(s).
        :rtype: RegistedType | list[RegistedType]
        """
        choice = getattr(config, f"{self.name}_type", None)
        if choice is None:
            return None
        if isinstance(choice, (list, ListConfig)):
            loaded = []
            for name in choice:
                if name in self:
                    cfg_name = f"{self[name]['short_names'][0]}_config"
                    sub_cfg = getattr(config, cfg_name, None)
                    if sub_cfg is None:
                        loaded.append(self[name]["item"](**kwargs))
                    else:
                        loaded.append(self[name]["item"](sub_cfg, **kwargs))
        elif choice in self:
            cfg_name = f"{self[choice]['short_names'][0]}_config"
            sub_cfg = getattr(config, cfg_name, None)
            if sub_cfg is None:
                loaded = self[choice]["item"](**kwargs)
            else:
                loaded = self[choice]["item"](sub_cfg, **kwargs)
        else:
            raise ValueError(f"Invalid {self.name} type: {choice}")
        return loaded

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

    def __pickle__(self):
        return self.value


def Choices(choices: Iterable[str]):
    return StrEnum("Choices", {c: c for c in choices})


# Monkey Patching the JSONEncoder to handle StrEnum
class _CustomEncoder(json.JSONEncoder):
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


json.dumps = partial(json.dumps, cls=_CustomEncoder)
json.dump = partial(json.dump, cls=_CustomEncoder)


class _TimeMeter:
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

            async def async_wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = await func(*args, **kwargs)
                end_time = perf_counter()
                if timer_names not in self.timers:
                    self.timers[timer_names] = self._manager.list()
                self.timers[timer_names].append(end_time - start_time)
                return result

            if isinstance(func, Coroutine):
                return async_wrapper
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


TIME_METER = _TimeMeter()


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, color_map: dict[str, str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if color_map is None:
            color_map = {
                "DEBUG": colorama.Fore.CYAN,
                "INFO": colorama.Fore.GREEN,
                "WARNING": colorama.Fore.YELLOW,
                "ERROR": colorama.Fore.RED,
                "CRITICAL": colorama.Fore.RED,
            }
        self.color_map = color_map
        return

    def format(self, record) -> str:
        message = super().format(record)
        color = self.color_map.get(record.levelname, "")
        levelname = record.levelname
        colored_levelname = f"{color}{levelname}{colorama.Style.RESET_ALL}"
        message = message.replace(levelname, colored_levelname)
        return message


class _LoggerManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if not cls._instance:
            with cls._lock:  # ensure thread safety
                if not cls._instance:
                    cls._instance = super(_LoggerManager, cls).__new__(cls)
                    cls._instance._configure()  # initialize the LoggerManager
        return cls._instance

    def _configure(self):
        self.loggers: dict[str, logging.Logger] = {}
        self.default_level = os.environ.get("LOGLEVEL", "INFO")
        self.default_fmt = ColoredFormatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.default_handler = logging.StreamHandler()
        self.default_handler.setLevel(self.default_level)
        self.default_handler.setFormatter(self.default_fmt)
        return

    def getLogger(self, name: str) -> logging.Logger:
        """Get the logger by name. If the logger does not exist, create a new one.

        :param name: The name of the logger.
        :type name: str
        :return: The logger.
        :rtype: logging.Logger
        """
        return self.get_logger(name)

    def get_logger(self, name: str) -> logging.Logger:
        """Get the logger by name. If the logger does not exist, create a new one.

        :param name: The name of the logger.
        :type name: str
        :return: The logger.
        :rtype: logging.Logger
        """
        if name not in self.loggers:
            self.loggers[name] = logging.getLogger(name)
            self.loggers[name].propagate = False  # prevent duplicate logs
            self.add_handler(self.default_handler, name)
            self.set_level(self.default_level, name)
        return self.loggers[name]

    def add_handler(self, handler: logging.Handler, name: str = None):
        """Add the handler to the logger.

        :param handler: The handler to add.
        :type handler: logging.Handler
        :param name: The name of the logger, None for all FlexRAG loggers, defaults to None.
        :type name: str, optional
        """
        if name is None:
            for logger in self.loggers.values():
                logger.addHandler(handler)
        else:
            logger = self.get_logger(name)
            logger.addHandler(handler)
        return

    def remove_handler(self, handler: logging.Handler, name: str = None):
        """Remove the handler from the logger.

        :param handler: The handler to remove.
        :type handler: logging.Handler
        :param name: The name of the logger, None for all FlexRAG loggers, defaults to None.
        :type name: str, optional
        """
        if name is None:
            for logger in self.loggers.values():
                logger.removeHandler(handler)
        else:
            logger = self.get_logger(name)
            logger.removeHandler(handler)
        return

    def set_level(self, level: int, name: str = None):
        """Set the level of the logger.

        :param level: The level to set.
        :type level: int
        :param name: The name of the logger, None for all FlexRAG loggers, defaults to None.
        :type name: str, optional
        """
        if name is None:
            for logger in self.loggers.values():
                logger.setLevel(level)
        else:
            logger = self.get_logger(name)
            logger.setLevel(level)
        return

    def set_formatter(self, formatter: logging.Formatter | str, name: str = None):
        """Set the formatter of the logger.

        :param formatter: The formatter to set.
        :type formatter: logging.Formatter | str
        :param name: The name of the logger, None for all FlexRAG loggers, defaults to None.
        :type name: str, optional
        """
        if isinstance(formatter, str):
            formatter = logging.Formatter(formatter)
        if name is None:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    handler.setFormatter(formatter)
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
        else:
            logger = self.get_logger(name)
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        return


LOGGER_MANAGER = _LoggerManager()


def load_user_module(module_path: str):
    module_path = os.path.abspath(module_path)
    module_parent, module_name = os.path.split(module_path)
    if module_name not in sys.modules:
        sys.path.insert(0, module_parent)
        importlib.import_module(module_name)
