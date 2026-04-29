from .guten_qa import GutenQATask, GutenQATaskConfig
from .literary_qa import LiteraryQATask, LiteraryQATaskConfig
from .longbench import LongBenchTask, LongBenchTaskConfig
from .multihop_rag import MultihopRAGTask, MultihopRAGTaskConfig
from .narrative_qa import NarrativeQATask, NarrativeQATaskConfig
from .squad import SQuADTask, SQuADTaskConfig

__all__ = [
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
]
