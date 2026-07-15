from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import MultihopRAGDataset, MultihopRAGDatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import (
    F1,
    Accuracy,
    AccuracyConfig,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
)

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


@configure
class MultihopRAGTaskConfig(ContextualQATaskConfig, MultihopRAGDatasetConfig):
    """Configuration for MultihopRAG Task."""


@TASKS("multihop_rag", config_class=MultihopRAGTaskConfig)
class MultihopRAGTask(ContextualQATask):
    """Contextualized QA Task on Multihop RAG dataset."""

    instruction = (
        "Below is a question followed by some context from different sources. Please"
        " answer the question based on the context. The answer to the question is a"
        " word or entity. If the provided information is insufficient to answer the"
        " question, respond 'Insufficient Information'. Answer directly without"
        " explanation.\n\nQuestion:{question}\n\nContext:{context}"
    )

    def load_dataset(self) -> MultihopRAGDataset:
        return MultihopRAGDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "accuracy": Accuracy(AccuracyConfig()),
        }
        return Evaluator(metrics)

    async def evaluate(
        self, assistant: AssistantProtocol, sample: ContextualQASample
    ) -> AssistantResult:
        context_text = ""
        for context in sample.contexts:
            context_text += context.data["text"] + "\n"
        context_text = context_text.strip()
        # construct the prompt
        prompt = self.instruction.format(context=context_text, question=sample.question)
        response = await assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
        return response
