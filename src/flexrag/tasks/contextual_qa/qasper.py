import re
import string
from collections import Counter
from typing import Any

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.datasets.benchmarks import QasperDataset, QasperDatasetConfig
from flexrag.datasets.core import ContextualQASample
from flexrag.metrics import Evaluator

from ..contextual_qa_base import ContextualQATask, ContextualQATaskConfig
from ..task_base import TASKS


@configure
class QasperTaskConfig(ContextualQATaskConfig, QasperDatasetConfig):
    """Configuration for Qasper Task."""


class _QasperAnswerF1:
    """Official-style answer F1 metric for Qasper."""

    answer_types = ("extractive", "abstractive", "boolean", "none")

    def normalize_answer(self, text: str) -> str:
        def remove_articles(text: str) -> str:
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def remove_punc(text: str) -> str:
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        text = text.lower()
        text = remove_punc(text)
        text = remove_articles(text)
        return " ".join(text.split())

    def token_f1_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = num_same / len(prediction_tokens)
        recall = num_same / len(ground_truth_tokens)
        return (2 * precision * recall) / (precision + recall)

    def build_references(self, metadata: dict[str, Any]) -> list[dict[str, str]]:
        references = []
        for annotation in metadata.get("annotations", []):
            answer = annotation.get("answer", {})
            if answer.get("unanswerable"):
                references.append({"answer": "Unanswerable", "type": "none"})
            elif answer.get("extractive_spans", []):
                references.append(
                    {
                        "answer": ", ".join(answer["extractive_spans"]),
                        "type": "extractive",
                    }
                )
            elif answer.get("free_form_answer"):
                references.append(
                    {
                        "answer": answer["free_form_answer"],
                        "type": "abstractive",
                    }
                )
            elif answer.get("yes_no") is True:
                references.append({"answer": "Yes", "type": "boolean"})
            elif answer.get("yes_no") is False:
                references.append({"answer": "No", "type": "boolean"})
        return references

    def __call__(
        self, responses: list[str], metadatas: list[dict[str, Any]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        item_scores = []
        item_types = []
        scores_by_type = {answer_type: [] for answer_type in self.answer_types}

        for response, metadata in zip(responses, metadatas):
            references = self.build_references(metadata)
            if not references:
                item_scores.append(0.0)
                item_types.append("missing")
                continue

            f1s_and_types = [
                (self.token_f1_score(response, reference["answer"]), reference["type"])
                for reference in references
            ]
            max_f1, answer_type = sorted(f1s_and_types, key=lambda x: x[0])[-1]
            item_scores.append(max_f1)
            item_types.append(answer_type)
            scores_by_type[answer_type].append(max_f1)

        def mean(scores: list[float]) -> float:
            return sum(scores) / len(scores) if scores else 0.0

        scores = {
            "answer_f1": mean(item_scores),
            "extractive_f1": mean(scores_by_type["extractive"]),
            "abstractive_f1": mean(scores_by_type["abstractive"]),
            "boolean_f1": mean(scores_by_type["boolean"]),
            "none_f1": mean(scores_by_type["none"]),
        }
        details = {
            "item_score": item_scores,
            "item_type": item_types,
            "answer_f1_by_type": {
                answer_type: mean(scores)
                for answer_type, scores in scores_by_type.items()
            },
        }
        return scores, details


@TASKS("qasper", config_class=QasperTaskConfig)
class QasperTask(ContextualQATask):
    """Contextualized QA Task on Qasper dataset."""

    instruction = (
        "You are given a scientific article and a question. Answer the question as"
        " concisely as you can, using a single phrase or sentence if possible. If"
        " the question cannot be answered based on the information in the article,"
        ' write "unanswerable". If the question is a yes/no question, answer "yes",'
        ' "no", or "unanswerable". Do not provide any explanation.\n\nArticle:\n'
        "{context}\n\nQuestion:\n{question}"
    )

    def load_dataset(self) -> QasperDataset:
        return QasperDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        return Evaluator({"qasper_answer_f1": _QasperAnswerF1()})

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualQASample
    ) -> AssistantResponse:
        context = "\n".join(context.data["text"] for context in sample.contexts)
        prompt = self.instruction.format(context=context, question=sample.question)
        return assistant.answer(
            messages=ChatMessages.from_list([ChatTurn(role="user", content=prompt)])
        )
