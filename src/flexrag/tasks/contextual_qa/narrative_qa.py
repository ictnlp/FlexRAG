from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import NarrativeQADataset, NarrativeQADatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import (
    F1,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


@configure
class NarrativeQATaskConfig(ContextualQATaskConfig, NarrativeQADatasetConfig):
    """Configuration for NarrativeQA Task."""


@TASKS("narrative_qa", config_class=NarrativeQATaskConfig)
class NarrativeQATask(ContextualQATask):
    """Contextualized QA Task on NarrativeQA dataset."""

    instruction = (
        "You are given a narrative context and a question.\n\nAnswer the question using"
        " only the information provided in the context.\nIf the answer cannot be"
        ' determined from the context, answer "Not answerable".\n\nContext:\n{context}'
        "\n\nQuestion:\n{question}"
    )

    def load_dataset(self) -> NarrativeQADataset:
        return NarrativeQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "rouge": Rouge(RougeConfig()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        # construct the prompt
        prompt = self.instruction.format(
            context=sample.contexts[0].data["text"], question=sample.question
        )
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response
