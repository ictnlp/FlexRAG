import re
from typing import Any

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import configure
from flexrag.datasets.benchmarks import LongBenchV2Dataset, LongBenchV2DatasetConfig
from flexrag.datasets.core import ContextualMCSample, MappingDataset
from flexrag.metrics import Evaluator

from ..contextual_mc_base import ContextualMCTask, ContextualMCTaskConfig
from ..task_base import TASKS


@configure
class LongBenchV2TaskConfig(ContextualMCTaskConfig, LongBenchV2DatasetConfig):
    """Configuration for LongBenchV2 Contextual Multiple Choice Task."""


class _LongBenchV2Metric:
    def extract_answer(self, response: str) -> str:
        response = response.replace("*", "").lower()
        matches = re.findall(r"the correct answer is\s*[ (]*([a-d])", response)
        return matches[-1] if matches else None

    def __call__(
        self, responses: list[str], golden_responses: list[list[int]]
    ) -> tuple[dict[str, float], dict[str, Any]]:
        correctness = []
        for resp, golden in zip(responses, golden_responses):
            pred = self.extract_answer(resp)
            pred = ord(pred) - ord("a") if pred is not None else None
            if pred in golden:
                correctness.append(1.0)
            else:
                correctness.append(0.0)
        final_score = sum(correctness) / len(correctness) if correctness else 0.0
        return {"accuracy": final_score}, {"correctness": correctness}


@TASKS("longbench_v2", config_class=LongBenchV2TaskConfig)
class LongBenchV2Task(ContextualMCTask):
    """Contextual Multiple Choice Task for LongBenchV2 dataset."""

    instruct = (
        "Please read the following text and answer the question below.\n\n<text>"
        "\n{context}\n</text>\n\nWhat is the correct answer to this question:"
        " {question}\nChoices:\n(A) {A}\n(B) {B}\n(C) {C}\n(D) {D}\n\nFormat your"
        ' response as follows: "The correct answer is (insert answer here)".'
    )

    def load_dataset(self) -> MappingDataset[ContextualMCSample]:
        return LongBenchV2Dataset(self.config)

    def load_evaluator(self) -> Evaluator:
        return Evaluator({"longbench_v2_metric": _LongBenchV2Metric()})

    def evaluate(
        self, assistant: AssistantBase, sample: ContextualMCSample
    ) -> AssistantResponse:
        # construct the question with retrieved contexts
        prompt = self.instruct.format(
            question=sample.question,
            A=sample.choices[0],
            B=sample.choices[1],
            C=sample.choices[2],
            D=sample.choices[3],
            context=sample.contexts[0].data["text"],
        )
        # get response from assistant
        response = assistant.answer([{"role": "user", "content": prompt}])
        return response
