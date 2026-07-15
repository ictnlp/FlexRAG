import asyncio
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantProtocol, AssistantResult
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
    """Configuration for multi-session QA evaluation.

    :param log_interval: Progress logging interval.
    :param output_path: Optional directory for evaluation outputs.
    """

    log_interval: int = 10
    output_path: Optional[str] = None


class MultiSessionQATask(TaskBase):
    """Base class for episode-isolated multi-session QA tasks."""

    config: MultiSessionQATaskConfig

    def setup(self) -> None:
        """Load the dataset, evaluator, logging, and output paths."""
        self.logger = LOGGER_MANAGER.get_logger("task.multi_session_qa")
        if self.config.output_path is not None:
            os.makedirs(self.config.output_path, exist_ok=True)
            LOGGER_MANAGER.add_handler(
                logging.FileHandler(Path(self.config.output_path, "log.txt"))
            )
        self.logger.debug(f"Configs:\n{self.config.dumps()}")

        if self.config.output_path is not None:
            self.details_path = Path(self.config.output_path, "details.jsonl")
            self.eval_score_path = Path(self.config.output_path, "eval_score.json")
            self.config_path = Path(self.config.output_path, "config.json")
        else:
            self.details_path = Path(os.devnull)
            self.eval_score_path = Path(os.devnull)
            self.config_path = Path(os.devnull)
        self.config.dump(self.config_path)
        self.testset = self.load_dataset()
        self.evaluator = self.load_evaluator()

    async def run(
        self,
        assistant_factory: Callable[[], AssistantProtocol],
    ) -> None:
        """Evaluate groups serially and questions within each group concurrently.

        :param assistant_factory: Factory returning a fresh assistant for each
            context group.
        """
        groups: dict[str | None, list[MultiSessionQASample]] = defaultdict(list)
        for item in self.testset:
            groups[item.sessions_id].append(item)

        questions: list[str] = []
        golden_answers: list[list[str] | None] = []
        responses: list[str] = []
        contexts: list[list[RetrievedContext]] = []
        metadatas: list[dict] = []
        details: list[tuple[MultiSessionQASample, AssistantResult]] = []
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(self.testset)
        )

        for group in groups.values():
            async with assistant_factory() as assistant:
                await assistant.add_histories(group[0].sessions)

                async def evaluate_one(
                    item: MultiSessionQASample,
                ) -> AssistantResult:
                    result = await self.evaluate(assistant=assistant, sample=item)
                    p_logger.update(desc="Inferencing")
                    return result

                group_results = await asyncio.gather(
                    *(evaluate_one(item) for item in group)
                )

            for item, result in zip(group, group_results, strict=True):
                questions.append(item.question)
                golden_answers.append(item.answers)
                responses.append(result.response.text_content or "")
                contexts.append(result.contexts)
                metadatas.append(item.metadata or {})
                details.append((item, result))

        with open(self.details_path, "w", encoding="utf-8") as f:
            for item, result in details:
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden": item.answers,
                            "metadata_test": item.metadata,
                            "response": result,
                        },
                        to_bytes=False,
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

        resp_score, resp_score_detail = await asyncio.to_thread(
            self.evaluator.evaluate,
            questions=questions,
            responses=responses,
            golden_responses=golden_answers,
            retrieved_contexts=contexts,
            metadata=metadatas,
            log=True,
        )
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

    @abstractmethod
    async def evaluate(
        self,
        assistant: AssistantProtocol,
        sample: MultiSessionQASample,
    ) -> AssistantResult:
        """Evaluate one multi-session QA sample.

        :param assistant: Active assistant episode initialized for the group.
        :param sample: Multi-session QA sample.
        :return: Assistant result for the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self) -> MappingDataset[MultiSessionQASample]:
        """Load the multi-session QA dataset.

        :return: Multi-session QA dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the response evaluator.

        :return: Configured evaluator.
        """
        raise NotImplementedError
