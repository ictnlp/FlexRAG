from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import GutenQADataset, GutenQADatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import (
    F1,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
)

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


@configure
class GutenQATaskConfig(ContextualQATaskConfig, GutenQADatasetConfig):
    """Configuration for GutenQA Task."""


@TASKS("guten_qa", config_class=GutenQATaskConfig)
class GutenQATask(ContextualQATask):
    """Contextualized QA Task on GutenQA dataset."""

    instructions = {
        "book": (
            "You are given a question and the complete text of a book. Answer the"
            " question based on the information in the book. If the answer cannot be"
            ' determined from the book, output "Insufficient information".\n\n'
            "Question:\n{question}\n\nBook:\n{context}\n\nReturn only the final answer"
            " text, with no extra commentary."
        ),
        "chunk": (
            "You are given a question and several context chunks extracted from a book."
            " Answer the question based on the information in the context chunks. If"
            ' the answer cannot be determined from the contexts, output "Insufficient'
            ' information".\n\nQuestion:\n{question}\n\nContexts:\n{context}\n\nReturn'
            " only the final answer text, with no extra commentary."
        ),
    }

    def load_dataset(self) -> GutenQADataset:
        return GutenQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        if self.config.context_mode == "book":
            context_text = sample.contexts[0].data["text"]
            template = self.instructions["book"]
        else:
            context_text = ""
            for context in sample.contexts:
                context_text += context.data["text"] + "\n"
            context_text = context_text.strip()
            template = self.instructions["chunk"]
        # construct the prompt
        prompt = template.format(context=context_text, question=sample.question)
        response = assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response
