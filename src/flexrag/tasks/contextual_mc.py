import logging
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import (
    LOGGER_MANAGER,
    ChatMessages,
    RetrievedContext,
    SimpleProgressLogger,
    configure,
)
from flexrag.common.database import json_dump
from flexrag.datasets.benchmarks import LongBenchV2Dataset, LongBenchV2DatasetConfig
from flexrag.datasets.core import ContextualMCSample, MappingDataset
from flexrag.metrics import Evaluator

from .task_base import TASKS, TaskBase


@configure
class ContextualMCTaskConfig:
    """Configuration for Contextual Multiple Choice Task."""

    log_interval: int = 10
    output_path: Optional[str] = None


class ContextualMCTask(TaskBase):
    """Base class for all Contextual Multiple Choice Tasks."""

    config: ContextualMCTaskConfig

    def setup(self):
        """Setup the Contextual Multiple Choice task."""
        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.contextual_mc")
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            log_path = Path(self.config.output_path, "log.txt")
            handler = logging.FileHandler(log_path)
            LOGGER_MANAGER.add_handler(handler)
        self.logger.debug(f"Configs:\n{self.config.dumps()}")

        # setup output paths
        if self.config.output_path is not None:
            self.details_path = Path(self.config.output_path, "details.jsonl")
            self.eval_score_path = Path(self.config.output_path, "eval_score.json")
            self.config_path = Path(self.config.output_path, "config.json")
        else:
            self.details_path = Path(os.devnull)
            self.eval_score_path = Path(os.devnull)
            self.config_path = Path(os.devnull)
        self.config.dump(self.config_path)

        # load dataset
        self.testset = self.load_dataset()

        # load metrics
        self.evaluator = self.load_evaluator()
        return

    def run(self, assistant: AssistantBase):
        """Run the Contextual Multiple Choice task."""
        # search and answer questions
        questions: list[str] = []
        golden_answers: list[list[str]] = []
        responses: list[str] = []
        contexts: list[list[RetrievedContext]] = []
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(self.testset)
        )
        with open(self.details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                questions.append(item.question)
                golden_answers.append(item.answers)
                response = self.evaluate(assistant=assistant, sample=item)
                responses.append(response.response.text_content)
                contexts.append(response.contexts)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden": item.answers,
                            "metadata_test": item.meta_data,
                            "response": response,
                        },
                        to_bytes=False,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                p_logger.update(desc="Inferencing")

        # Evaluate the results
        resp_score, resp_score_detail = self.evaluator.evaluate(
            questions=questions,
            responses=responses,
            golden_responses=golden_answers,
            golden_contexts=contexts,
            log=True,
        )

        # Save the evaluation results
        with open(self.eval_score_path, "w", encoding="utf-8") as f:
            f.write(
                json_dump(
                    {
                        "eval_scores": resp_score,
                        "eval_details": resp_score_detail,
                    },
                    to_bytes=False,
                    indent=4,
                    ensure_ascii=False,
                )
            )
        return

    @abstractmethod
    def evaluate(
        self, assistant: AssistantBase, sample: ContextualMCSample
    ) -> AssistantResponse:
        """
        Evaluate a single data sample.

        :param assistant: The assistant to evaluate.
        :type assistant: AssistantBase
        :param sample: A single data sample to be evaluated.
        :type sample: ContextualMCSample
        :return: The response from the assistant.
        :rtype: AssistantResponse
        """
        return

    @abstractmethod
    def load_dataset(self) -> MappingDataset[ContextualMCSample]:
        """Load the dataset for the task.

        :return: The dataset for the task.
        :rtype: MappingDataset[ContextualMCSample]
        """
        return

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the evaluator for the task.

        :return: The evaluator for the task.
        :rtype: Evaluator
        """
        return


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
