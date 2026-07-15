import asyncio
import logging
import os
from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import LOGGER_MANAGER, SimpleProgressLogger, configure
from flexrag.common.dataclasses import Context
from flexrag.common.serialization import json_dump
from flexrag.datasets.core import ContextualQASample, MappingDataset
from flexrag.metrics import Evaluator

from .task_base import TaskBase


@configure
class ContextualQATaskConfig:
    """Configuration for contextual QA evaluation.

    :param log_interval: Progress logging interval.
    :param output_path: Optional directory for evaluation outputs.
    """

    log_interval: int = 10
    output_path: Optional[str] = None


class ContextualQATask(TaskBase):
    """Base class for contextual QA tasks."""

    config: ContextualQATaskConfig

    def setup(self) -> None:
        """Load the dataset, evaluator, logging, and output paths."""
        self.logger = LOGGER_MANAGER.get_logger("task.contextualized_qa")
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
        """Evaluate all samples concurrently in one assistant episode.

        Existing benchmark prompts remain responsible for rendering their
        provided contexts.

        :param assistant_factory: Factory returning a fresh assistant instance.
        """
        items = list(self.testset)
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(items)
        )
        async with assistant_factory() as assistant:

            async def evaluate_one(item: ContextualQASample) -> AssistantResult:
                result = await self.evaluate(assistant=assistant, sample=item)
                p_logger.update(desc="Inferencing")
                return result

            results = await asyncio.gather(*(evaluate_one(item) for item in items))

        questions = [item.question for item in items]
        golden_answers = [item.answers for item in items]
        metadatas = [item.metadata or {} for item in items]
        responses = [result.response.text_content or "" for result in results]
        contexts: list[list[Context]] = [item.contexts for item in items]

        with open(self.details_path, "w", encoding="utf-8") as f:
            for item, result in zip(items, results, strict=True):
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
            metadatas=metadatas,
            golden_contexts=contexts,
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
        sample: ContextualQASample,
    ) -> AssistantResult:
        """Evaluate one contextual QA sample.

        :param assistant: Active assistant episode.
        :param sample: Contextual QA sample.
        :return: Assistant result for the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self) -> MappingDataset[ContextualQASample]:
        """Load the contextual QA dataset.

        :return: Contextual QA dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the response evaluator.

        :return: Configured evaluator.
        """
        raise NotImplementedError
