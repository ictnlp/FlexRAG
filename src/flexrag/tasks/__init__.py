from .contextual_mc import (
    ContextualMCTask,
    ContextualMCTaskConfig,
    LongBenchV2Task,
    LongBenchV2TaskConfig,
)
from .contextual_qa import (
    ContextualQATask,
    ContextualQATaskConfig,
    GutenQATask,
    GutenQATaskConfig,
    LiteraryQATask,
    LiteraryQATaskConfig,
    LongBenchTask,
    LongBenchTaskConfig,
    MultihopRAGTask,
    MultihopRAGTaskConfig,
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
from .retrieval_task import (
    MLDRRetrievalTask,
    MLDRRetrievalTaskConfig,
    MSMARCORetrievalTask,
    MSMARCORetrievalTaskConfig,
    MTEBRetrievalTask,
    MTEBRetrievalTaskConfig,
    RetrievalTask,
    RetrievalTaskConfig,
)
from .task_base import TASKS, TaskBase, TaskBaseConfig

TaskConfig = TASKS.make_config()


__all__ = [
    "TaskBase",
    "TaskBaseConfig",
    "TASKS",
    "TaskConfig",
    "MTEBRetrievalTask",
    "MTEBRetrievalTaskConfig",
    "RetrievalTask",
    "RetrievalTaskConfig",
    "ContextualQATask",
    "ContextualQATaskConfig",
    "GutenQATask",
    "GutenQATaskConfig",
    "LiteraryQATask",
    "LiteraryQATaskConfig",
    "LongBenchTask",
    "LongBenchTaskConfig",
    "MultihopRAGTask",
    "MultihopRAGTaskConfig",
    "NarrativeQATask",
    "NarrativeQATaskConfig",
    "SQuADTask",
    "SQuADTaskConfig",
    "ContextualMCTask",
    "ContextualMCTaskConfig",
    "LongBenchV2Task",
    "LongBenchV2TaskConfig",
    "BrowseCompTask",
    "BrowseCompTaskConfig",
    "OpenQATask",
    "OpenQATaskConfig",
    "SimpleQATask",
    "SimpleQATaskConfig",
]
