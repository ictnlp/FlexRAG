from .contextual_mc import (
    LongBenchV2Task,
    LongBenchV2TaskConfig,
)
from .contextual_mc_base import ContextualMCTask, ContextualMCTaskConfig
from .contextual_qa import (
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
from .contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from .deep_search import (
    BrowseCompTask,
    BrowseCompTaskConfig,
    BrowseCompZHTask,
    BrowseCompZHTaskConfig,
    GISATask,
    GISATaskConfig,
    WideSearchTask,
    WideSearchTaskConfig,
)
from .file_qa import UDAQATask, UDAQATaskConfig
from .memory import (
    ConvoMemTask,
    ConvoMemTaskConfig,
    LoCoMoTask,
    LoCoMoTaskConfig,
    LongMemEvalTask,
    LongMemEvalTaskConfig,
)
from .multisession_qa_base import MultiSessionQATask, MultiSessionQATaskConfig
from .open_qa import (
    SimpleQATask,
    SimpleQATaskConfig,
)
from .open_qa_base import OpenQATask, OpenQATaskConfig
from .retrieval import (
    MLDRRetrievalTask,
    MLDRRetrievalTaskConfig,
    MSMARCORetrievalTask,
    MSMARCORetrievalTaskConfig,
    MTEBRetrievalTask,
    MTEBRetrievalTaskConfig,
)
from .retrieval_base import (
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
    "MLDRRetrievalTask",
    "MLDRRetrievalTaskConfig",
    "MSMARCORetrievalTask",
    "MSMARCORetrievalTaskConfig",
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
    "ConvoMemTask",
    "ConvoMemTaskConfig",
    "LoCoMoTask",
    "LoCoMoTaskConfig",
    "LongMemEvalTask",
    "LongMemEvalTaskConfig",
    "MultiSessionQATask",
    "MultiSessionQATaskConfig",
    "ContextualMCTask",
    "ContextualMCTaskConfig",
    "LongBenchV2Task",
    "LongBenchV2TaskConfig",
    "BrowseCompTask",
    "BrowseCompTaskConfig",
    "BrowseCompZHTask",
    "BrowseCompZHTaskConfig",
    "GISATask",
    "GISATaskConfig",
    "WideSearchTask",
    "WideSearchTaskConfig",
    "OpenQATask",
    "OpenQATaskConfig",
    "SimpleQATask",
    "SimpleQATaskConfig",
    "UDAQATask",
    "UDAQATaskConfig",
]
