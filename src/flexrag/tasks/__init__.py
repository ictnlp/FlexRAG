from .contextual_qa import (
    ContextualQATask,
    ContextualQATaskConfig,
    LongBenchTask,
    LongBenchTaskConfig,
    NarrativeQATask,
    NarrativeQATaskConfig,
    SQuADTask,
    SQuADTaskConfig,
)
from .open_qa import (
    BrowseCompTask,
    BrowseCompTaskConfig,
    OpenQATask,
    OpenQATaskConfig,
    SimpleQATask,
    SimpleQATaskConfig,
)
from .retrieval_task import RetrievalTask, RetrievalTaskConfig
from .task_base import TASKS, TaskBase, TaskBaseConfig

TaskConfig = TASKS.make_config()


__all__ = [
    "TaskBase",
    "TaskBaseConfig",
    "TASKS",
    "TaskConfig",
    "RetrievalTask",
    "RetrievalTaskConfig",
    "ContextualQATask",
    "ContextualQATaskConfig",
    "LongBenchTask",
    "LongBenchTaskConfig",
    "NarrativeQATask",
    "NarrativeQATaskConfig",
    "SQuADTask",
    "SQuADTaskConfig",
    "BrowseCompTask",
    "BrowseCompTaskConfig",
    "OpenQATask",
    "OpenQATaskConfig",
    "SimpleQATask",
    "SimpleQATaskConfig",
]
