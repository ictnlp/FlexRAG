import logging
import os
import platform
import threading
from time import perf_counter

import colorama

if platform.system() == "Windows":
    colorama.just_fix_windows_console()


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
