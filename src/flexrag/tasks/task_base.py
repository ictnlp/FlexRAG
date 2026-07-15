import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

from flexrag.common import LOGGER_MANAGER, Register, configure


@configure
class TaskBaseConfig:
    """Common configuration for evaluation tasks.

    :param log_interval: Progress logging interval.
    :param output_path: Optional directory for evaluation outputs.
    """

    log_interval: int = 10
    output_path: str | None = None


class TaskBase(ABC):
    """Common construction base for FlexRAG evaluation tasks.

    A task is ready to run after construction. Concrete tasks receive their
    evaluation target, or a factory for it, during construction and expose a
    uniform asynchronous execution entry point.
    """

    _logger_name: ClassVar[str] = "task"

    def __init__(self, config: TaskBaseConfig) -> None:
        """Initialize common logging and output paths.

        :param config: Common task configuration.
        """
        self.config = config
        self.logger = LOGGER_MANAGER.get_logger(self._logger_name)

        if config.output_path is not None:
            output_path = Path(config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            LOGGER_MANAGER.add_handler(logging.FileHandler(output_path / "log.txt"))
            self.details_path = output_path / "details.jsonl"
            self.eval_score_path = output_path / "eval_score.json"
            self.config_path = output_path / "config.json"
        else:
            self.details_path = Path(os.devnull)
            self.eval_score_path = Path(os.devnull)
            self.config_path = Path(os.devnull)

        self.logger.debug(f"Configs:\n{config.dumps()}")
        config.dump(self.config_path)

    @abstractmethod
    async def run(self) -> None:
        """Execute the configured evaluation task.

        Implementations evaluate the configured target and write their detail
        and score outputs.

        :return: None.
        """
        raise NotImplementedError


TASKS = Register[TaskBase](register_name="task")
