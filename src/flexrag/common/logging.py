import logging
import os
import platform
import threading
from time import perf_counter
from typing import Literal

import colorama
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if platform.system() == "Windows":
    colorama.just_fix_windows_console()


class SimpleProgressLogger:
    """Log coarse-grained progress updates through a standard logger.

    :param logger: The logger used to emit progress messages. If omitted,
        FlexRAG's default logger is used.
    :type logger: logging.Logger, optional
    :param total: The expected total number of updates. When provided, log
        messages include progress percentage and estimated remaining time.
    :type total: int, optional
    :param interval: The number of updates between automatic log messages.
        Set to ``0`` to disable automatic progress logging.
    :type interval: int
    :param display: The display mode for progress updates. Use ``"none"``
        to disable all output, ``"log"`` for coarse-grained logging,
        ``"bar"`` for a rich progress bar, or ``"auto"`` to choose
        ``"bar"`` for interactive terminals and ``"log"`` otherwise.
    :type display: Literal["none", "log", "bar", "auto"]
    """

    _DisplayMode = Literal["none", "log", "bar", "auto"]

    def __init__(
        self,
        logger: logging.Logger | None = None,
        total: int | None = None,
        interval: int = 100,
        display: _DisplayMode = "log",
    ):
        # set logger
        if logger is None:
            self.logger = LOGGER_MANAGER.default_logger
        else:
            self.logger = logger

        if display not in {"none", "log", "bar", "auto"}:
            raise ValueError(f"Unsupported display mode: {display}")

        # set arguments
        self.total = total
        self.interval = interval
        self.current = 0
        self.current_stage = 0
        self.desc = "Progress"
        self.start_time = perf_counter()
        self._lock = threading.RLock()
        self._closed = False
        self.display = display
        self._resolved_display = self._resolve_display_mode(display)
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        if self._resolved_display == "bar":
            self._init_progress()
        return

    def update(self, step: int = 1, desc: str | None = None) -> None:
        with self._lock:
            if self._closed:
                return
            if desc is not None:
                self.desc = desc
            self.current += step
            self._refresh_progress()
            if self._resolved_display == "none" or self.interval <= 0:
                return
            stage = self.current // self.interval
            if stage > self.current_stage:
                self.current_stage = stage
                self._log_progress()
        return

    def log(self) -> None:
        with self._lock:
            if self._closed or self._resolved_display == "none":
                return
            self._log_progress()
        return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._progress is not None:
                self._progress.stop()
                self._progress = None
                self._task_id = None
            if self._resolved_display != "none":
                self._log_progress()
            self._closed = True
        return

    def __enter__(self) -> "SimpleProgressLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        return

    def __repr__(self) -> str:
        return f"ProgressLogger({self.current}/{self.total})"

    def _resolve_display_mode(
        self, display: _DisplayMode
    ) -> Literal["none", "log", "bar"]:
        if display != "auto":
            return display
        return "bar" if LOGGER_MANAGER.is_interactive_console else "log"

    def _init_progress(self) -> None:
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
        if self.total is not None:
            columns.extend(
                [
                    BarColumn(),
                    MofNCompleteColumn(),
                    TaskProgressColumn(),
                    TimeElapsedColumn(),
                    TimeRemainingColumn(),
                ]
            )
        else:
            columns.extend(
                [
                    TextColumn("{task.completed:.0f}"),
                    TimeElapsedColumn(),
                ]
            )
        columns.append(TextColumn("{task.fields[speed]:.2f} update/s"))
        self._progress = Progress(
            *columns,
            console=LOGGER_MANAGER.console,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            self.desc,
            total=self.total,
            completed=0,
            speed=0.0,
        )
        return

    def _refresh_progress(self) -> None:
        if self._progress is None or self._task_id is None:
            return
        completed = self.current
        if self.total is not None:
            completed = min(completed, self.total)
        self._progress.update(
            self._task_id,
            description=self.desc,
            completed=completed,
            speed=self._get_speed(),
        )
        return

    def _log_progress(self) -> None:
        if (self.total is not None) and (self.current < self.total):
            time_spend = perf_counter() - self.start_time
            time_left = (
                time_spend * (self.total - self.current) / self.current
                if self.current > 0
                else 0.0
            )
            num_str = f"{self.current} / {self.total}"
            percent_str = f"({self.current/self.total:.2%})"
            time_str = (
                "["
                f"{self._fmt_time(time_spend)} / {self._fmt_time(time_left)}, "
                f"{self._get_speed():.2f} update/s"
                "]"
            )
            self.logger.info(f"{self.desc}: {num_str} {percent_str} {time_str}")
        else:
            time_spend = perf_counter() - self.start_time
            num_str = f"{self.current}"
            time_str = (
                f"[{self._fmt_time(time_spend)}, {self._get_speed():.2f} update/s]"
            )
            self.logger.info(f"{self.desc}: {num_str} {time_str}")
        return

    def _get_speed(self) -> float:
        elapsed = perf_counter() - self.start_time
        if elapsed <= 0:
            return 0.0
        return self.current / elapsed

    @staticmethod
    def _fmt_time(time: float) -> str:
        if time < 60:
            return f"{time:.2f}s"
        if time < 3600:
            return f"{time//60:02.0f}:{time%60:02.0f}"
        return f"{time//3600:.0f}:{(time%3600)//60:02.0f}:{time%60:02.0f}"


class ColoredFormatter(logging.Formatter):
    def __init__(self, *args, color_map: dict[str, str] | None = None, **kwargs):
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
        self.console = Console(stderr=True)
        self.default_fmt = logging.Formatter("%(name)s | %(message)s")
        self.default_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False,
            omit_repeated_times=False,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
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

    def add_handler(self, handler: logging.Handler, name: str | None = None):
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

    def remove_handler(self, handler: logging.Handler, name: str | None = None):
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

    def set_level(self, level: int | str, name: str | None = None):
        """Set the level of the logger.

        :param level: The level to set.
        :type level: int | str
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

    def set_formatter(
        self, formatter: logging.Formatter | str, name: str | None = None
    ):
        """Set the formatter of the logger.

        :param formatter: The formatter to set.
        :type formatter: logging.Formatter | str
        :param name: The name of the logger, None for all FlexRAG loggers, defaults to None.
        :type name: str, optional
        """
        if isinstance(formatter, str):
            formatter = logging.Formatter(formatter)
        self.default_fmt = formatter
        if name is None:
            for logger in self.loggers.values():
                for handler in logger.handlers:
                    handler.setFormatter(formatter)
        else:
            logger = self.get_logger(name)
            for handler in logger.handlers:
                handler.setFormatter(formatter)
        return

    @property
    def default_logger(self):
        return self.get_logger("flexrag")

    @property
    def is_interactive_console(self) -> bool:
        return bool(getattr(self.console, "is_terminal", False)) and not bool(
            getattr(self.console, "is_dumb_terminal", False)
        )


LOGGER_MANAGER = _LoggerManager()
