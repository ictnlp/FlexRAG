from abc import ABC, abstractmethod

from flexrag.utils import Register, configure


@configure
class TaskBaseConfig:
    """Configuration for TaskBase."""


class TaskBase(ABC):
    """Base class for all tasks in FlexRAG."""

    def __init__(self, config: TaskBaseConfig):
        self.config = config
        self.setup()
        return

    @abstractmethod
    def setup(self):
        """Setup the task."""
        return

    @abstractmethod
    def run(self):
        """Run the task."""
        return


TASKS = Register[TaskBase](register_name="task")
