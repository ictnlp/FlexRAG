from .qa_task import QATask, QATaskConfig
from .retrieval_task import RetrievalTask, RetrievalTaskConfig
from .tasks import TASKS, TaskBase, TaskBaseConfig

TaskConfig = TASKS.make_config()


__all__ = [
    "TaskBase",
    "TaskBaseConfig",
    "TASKS",
    "TaskConfig",
    "RetrievalTask",
    "RetrievalTaskConfig",
    "QATask",
    "QATaskConfig",
]
