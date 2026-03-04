from abc import ABC, abstractmethod

from flexrag.common import Register, configure


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
        """Setup the task.
        This method is used to initialize the task, such as loading the dataset, and
        initializing the metrics. The implementation of this method should be provided
        in the subclass. The `run` method will be called after the `setup` method, so
        the `setup` method can be used to prepare everything for the `run` method.
        """
        return

    @abstractmethod
    def run(self, assistant):
        """Run the task."""
        return


TASKS = Register[TaskBase](register_name="task")
