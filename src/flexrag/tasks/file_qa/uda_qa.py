import re
from collections import Counter

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.datasets.benchmarks import UDAQADataset, UDAQADatasetConfig
from flexrag.datasets.core import MappingDataset, QASample
from flexrag.metrics import (
    F1,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)
from flexrag.processors.text_processors import AnswerSimplifier

from ..open_qa_base import OpenQATask, OpenQATaskConfig
from ..task_base import TASKS


class _UDAQAOfficialMetric:
    """Subset-aware official metric wrapper for UDA-QA."""

    def __init__(self, subset: str):
        self.subset = subset
        self.scale_factors = {
            "": 1.0,
            "thousand": 1_000.0,
            "million": 1_000_000.0,
            "billion": 1_000_000_000.0,
            "percent": 1.0,
        }
        self.number_pattern = re.compile(r"\d[\d,]*(?:\.\d+)?")
        self.scale_pattern = re.compile(
            r"\b(?:thousand|thousands|million|millions|billion|billions|percent|bn|mn|k)\b|%"
        )
        self.simplifier = AnswerSimplifier()
        return

    def _token_f1(
        self, golds_tokens: list[list[str]], response_tokens: list[str]
    ) -> float:
        best_f1 = 0.0
        response_counter = Counter(response_tokens)
        for gold_tokens in golds_tokens:
            gold_counter = Counter(gold_tokens)
            common = sum((gold_counter & response_counter).values())
            if common == 0:
                continue
            precision = common / max(len(response_tokens), 1)
            recall = common / max(len(gold_tokens), 1)
            best_f1 = max(best_f1, (2 * precision * recall) / (precision + recall))
        return best_f1

    def _detect_scale_factor(self, context: str, default_scale: str) -> float:
        lowered = context.lower()
        if "%" in lowered or "percent" in lowered:
            return self.scale_factors["percent"]
        if "billion" in lowered or re.search(r"\bbn\b", lowered):
            return self.scale_factors["billion"]
        if "million" in lowered or re.search(r"\bmn\b", lowered):
            return self.scale_factors["million"]
        if "thousand" in lowered or re.search(r"\bk\b", lowered):
            return self.scale_factors["thousand"]
        return self.scale_factors.get(default_scale.lower(), 1.0)

    def _extract_numeric_values(
        self, text: str, default_scale: str = ""
    ) -> list[float]:
        lowered = str(text).lower()
        values = []
        for match in self.number_pattern.finditer(lowered):
            raw = match.group().replace(",", "")
            value = float(raw)
            prefix = lowered[max(0, match.start() - 6) : match.start()]
            suffix = lowered[match.end() : match.end() + 24]
            if "(" in prefix and ")" in suffix and not prefix.rstrip().endswith("-"):
                value = -value
            if prefix.rstrip().endswith("-"):
                value = -value
            factor = self._detect_scale_factor(f"{prefix} {suffix}", default_scale)
            values.append(value * factor)
        return values

    def _basic_f1_score(self, golds: list[str], response: str) -> float:
        golds_tokens = [self.simplifier(str(gold)).split() for gold in golds]
        response_tokens = self.simplifier(str(response)).split()
        return self._token_f1(golds_tokens, response_tokens)

    def _fin_exact_match(self, golds: list[str], response: str) -> float:
        response_values = self._extract_numeric_values(response)
        if len(response_values) == 1:
            for gold in golds:
                gold_values = self._extract_numeric_values(gold)
                if len(gold_values) != 1:
                    continue
                gold_value = gold_values[0]
                diff = abs(response_values[0] - gold_value)
                tolerance = abs(gold_value) * 0.01
                if gold_value == 0:
                    tolerance = 1e-9
                if diff <= tolerance:
                    return 1.0
        simplified_response = self.simplifier(str(response))
        return float(
            any(self.simplifier(str(gold)) == simplified_response for gold in golds)
        )

    def _tat_tokens(self, text: str, answer_scale: str) -> list[str]:
        values = [
            format(value, ".15g")
            for value in self._extract_numeric_values(text, answer_scale)
        ]
        text_without_numbers = self.number_pattern.sub(" ", str(text))
        text_without_scales = self.scale_pattern.sub(" ", text_without_numbers)
        text_tokens = self.simplifier(text_without_scales).split()
        return text_tokens + values

    def _tat_f1_score(self, golds: list[str], response: str, metadata: dict) -> float:
        answer_type = str(metadata.get("answer_type", ""))
        answer_scale = str(metadata.get("answer_scale", ""))
        response_tokens = self._tat_tokens(response, answer_scale)
        if answer_type == "multi-span":
            golds_tokens = [self._tat_tokens(" ".join(golds), answer_scale)]
        else:
            golds_tokens = [self._tat_tokens(gold, answer_scale) for gold in golds]
        return self._token_f1(golds_tokens, response_tokens)

    def __call__(
        self,
        responses: list[str],
        golden_responses: list[list[str]],
        metadatas: list[dict],
    ):
        scores = []
        for response, golds, metadata in zip(responses, golden_responses, metadatas):
            if self.subset in {"feta", "nq", "paper_text", "paper_tab"}:
                score = self._basic_f1_score(golds, response)
            elif self.subset == "fin":
                score = self._fin_exact_match(golds, response)
            else:
                score = self._tat_f1_score(golds, response, metadata)
            scores.append(score)
        official_score = sum(scores) / len(scores) if scores else 0.0
        return {"official_score": official_score}, {"item_score": scores}


@configure
class UDAQATaskConfig(OpenQATaskConfig, UDAQADatasetConfig):
    """Configuration for UDAQA Task."""


@TASKS("uda_qa", config_class=UDAQATaskConfig)
class UDAQATask(OpenQATask):
    """The UDA-QA Task for file-grounded question answering."""

    instruction = (
        "Read the attached document and answer the question. "
        "Return only the final answer without extra explanation.\n\n"
        "Question: {question}"
    )

    def load_dataset(self) -> MappingDataset[QASample]:
        return UDAQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        return Evaluator(
            {
                "f1": F1(F1Config()),
                "exact_match": ExactMatch(ExactMatchConfig()),
                "rouge": Rouge(RougeConfig()),
                "official_score": _UDAQAOfficialMetric(self.config.subset),
            }
        )

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        meta_data = sample.meta_data or {}
        file_path = meta_data["source_file_path"]
        file_format = meta_data["source_file_format"]
        prompt = self.instruction.format(question=sample.question)
        if file_format == "pdf":
            file_block = {"type": "pdf", "file_path": file_path}
        else:
            file_block = {
                "type": "file",
                "file_path": file_path,
                "mime_type": meta_data["source_mime_type"],
                "file_name": meta_data["source_file_name"],
            }
        return assistant.answer(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        file_block,
                    ],
                }
            ]
        )
