from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import configure
from flexrag.datasets.benchmarks import LoCoMoDataset, LoCoMoDatasetConfig
from flexrag.datasets.core import MultiSessionQASample
from flexrag.metrics import (
    F1,
    Accuracy,
    AccuracyConfig,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)

from ..multisession_qa_base import MultiSessionQATask, MultiSessionQATaskConfig
from ..task_base import TASKS


@configure
class LoCoMoTaskConfig(MultiSessionQATaskConfig, LoCoMoDatasetConfig):
    """Configuration for LoCoMo Task."""


@TASKS("locomo", config_class=LoCoMoTaskConfig)
class LoCoMoTask(MultiSessionQATask):
    """LoCoMo Task."""

    def load_dataset(self) -> LoCoMoDataset:
        return LoCoMoDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "em": ExactMatch(ExactMatchConfig()),
            "f1": F1(F1Config()),
            "rouge": Rouge(RougeConfig()),
            "accuracy": Accuracy(AccuracyConfig()),
        }
        return Evaluator(metrics)

    async def evaluate(
        self, assistant: AssistantProtocol, sample: MultiSessionQASample
    ) -> AssistantResult:
        return await assistant.answer([{"role": "user", "content": sample.question}])
