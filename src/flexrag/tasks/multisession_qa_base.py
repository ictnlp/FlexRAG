import logging
import os
from abc import abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import (
    LOGGER_MANAGER,
    RetrievedContext,
    SimpleProgressLogger,
    configure,
)
from flexrag.common.serialization import json_dump
from flexrag.datasets.core import MappingDataset, MultiSessionQASample
from flexrag.metrics import Evaluator

from .task_base import TaskBase


@configure
class MultiSessionQATaskConfig:
    """Configuration for Multi-Session QA Task.

    :param log_interval: The interval for logging progress during evaluation.
        Default is 10.
    :type log_interval: int
    :param output_path: The path to save the evaluation results and logs.
        If not specified results and logs will not be saved. Default is None.
    :type output_path: Optional[str]
    """

    log_interval: int = 10
    output_path: Optional[str] = None


class MultiSessionQATask(TaskBase):
    """Base class for all Multi-Session QA Tasks."""

    config: MultiSessionQATaskConfig

    def setup(self):
        """Setup the Multi-Session QA task."""
        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.multi_session_qa")
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
        """Run the Multi-Session QA task."""
        # group QA pairs by conversation sessions
        groups: dict[str, list[MultiSessionQASample]] = defaultdict(list)
        for item in self.testset:
            group_id = item.sessions_id
            groups[group_id].append(item)

        # search and answer questions
        questions: list[str] = []
        golden_answers: list[list[str]] = []
        responses: list[str] = []
        contexts: list[list[RetrievedContext]] = []
        metadatas: list[dict] = []
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(self.testset)
        )
        with open(self.details_path, "w", encoding="utf-8") as f:
            for group in groups.values():
                assistant.clear_histories()
                assistant.add_histories(group[0].sessions)
                for item in group:
                    questions.append(item.question)
                    golden_answers.append(item.answers)
                    response = self.evaluate(assistant=assistant, sample=item)
                    responses.append(response.response.text_content or "")
                    contexts.append(response.contexts or [])
                    metadatas.append(item.metadata)
                    f.write(
                        json_dump(
                            {
                                "question": item.question,
                                "golden": item.answers,
                                "metadata_test": item.metadata,
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
            retrieved_contexts=contexts,
            metadata=metadatas,
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
        self, assistant: AssistantBase, sample: MultiSessionQASample
    ) -> AssistantResponse:
        """
        Evaluate a single data sample.

        :param assistant: The assistant to evaluate.
        :type assistant: AssistantBase
        :param sample: A single data sample to be evaluated.
        :type sample: MultiSessionQASample
        :return: The response from the assistant.
        :rtype: AssistantResponse
        """
        return

    @abstractmethod
    def load_dataset(self) -> MappingDataset[MultiSessionQASample]:
        """Load the dataset for the task.

        :return: The dataset for the task.
        :rtype: MappingDataset[MultiSessionQASample]
        """
        return

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the evaluator for the task.

        :return: The evaluator for the task.
        :rtype: Evaluator
        """
        return
