from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import SQuADDataset, SQuADDatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import F1, Evaluator, ExactMatch, ExactMatchConfig, F1Config

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


@configure
class SQuADTaskConfig(ContextualQATaskConfig, SQuADDatasetConfig):
    """Configuration for SQuAD Task."""


@TASKS("squad", config_class=SQuADTaskConfig)
class SQuADTask(ContextualQATask):
    """Contextualized QA Task on SQuAD dataset."""

    instructions = {
        "v1.1": (
            "Read the following passage and answer the question.\nThe answer must be a"
            " span from the passage.\n\nPassage:\n{context}\n\nQuestion:\n{question}"
        ),
        "v2.0": (
            "Read the following passage and answer the question.\nIf the answer is not"
            ' contained in the passage, output "No Answer".\nOtherwise the answer must'
            " be an exact span from the passage.\n\nPassage:\n{context}\n\nQuestion:"
            "\n{question}"
        ),
    }

    def load_dataset(self) -> SQuADDataset:
        return SQuADDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        # construct the prompt
        prompt = self.instructions[self.config.version].format(
            context=sample.contexts[0].data["text"], question=sample.question
        )
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response
