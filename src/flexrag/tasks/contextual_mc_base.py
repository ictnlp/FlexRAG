import asyncio
from abc import abstractmethod
from collections.abc import Callable

from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import Context, SimpleProgressLogger, configure
from flexrag.common.serialization import json_dumps
from flexrag.datasets.core import ContextualMCSample, MappingDataset
from flexrag.metrics import Evaluator

from .task_base import TaskBase, TaskBaseConfig


@configure
class ContextualMCTaskConfig(TaskBaseConfig):
    """Configuration for contextual multiple-choice evaluation.

    :param log_interval: Progress logging interval.
    :param output_path: Optional directory for evaluation outputs.
    """


class ContextualMCTask(TaskBase):
    """Base class for contextual multiple-choice tasks."""

    config: ContextualMCTaskConfig
    _logger_name = "task.contextual_mc"

    def __init__(
        self,
        config: ContextualMCTaskConfig,
        *,
        assistant_factory: Callable[[], AssistantProtocol],
    ) -> None:
        """Load the dataset and evaluator for a contextual MC task.

        :param config: Contextual multiple-choice task configuration.
        :param assistant_factory: Factory returning a fresh assistant instance.
        """
        super().__init__(config)
        self._assistant_factory = assistant_factory
        self.testset = self.load_dataset()
        self.evaluator = self.load_evaluator()

    async def run(self) -> None:
        """Evaluate all samples concurrently in one assistant episode."""
        items = list(self.testset)
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(items)
        )
        async with self._assistant_factory() as assistant:

            async def evaluate_one(item: ContextualMCSample) -> AssistantResult:
                result = await self.evaluate(assistant=assistant, sample=item)
                p_logger.update(desc="Inferencing")
                return result

            results = await asyncio.gather(*(evaluate_one(item) for item in items))

        questions = [item.question for item in items]
        golden_answers = [item.answers for item in items]
        responses = [result.response.text_content or "" for result in results]
        contexts: list[list[Context]] = [item.contexts for item in items]

        with open(self.details_path, "w", encoding="utf-8") as f:
            for item, result in zip(items, results, strict=True):
                f.write(
                    json_dumps(
                        {
                            "question": item.question,
                            "golden": item.answers,
                            "metadata_test": item.metadata,
                            "response": result,
                        },
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

        resp_score, resp_score_detail = await asyncio.to_thread(
            self.evaluator.evaluate,
            questions=questions,
            responses=responses,
            golden_responses=golden_answers,
            golden_contexts=contexts,
            log=True,
        )
        with open(self.eval_score_path, "w", encoding="utf-8") as f:
            f.write(
                json_dumps(
                    {
                        "eval_scores": resp_score,
                        "eval_details": resp_score_detail,
                    },
                    indent=4,
                    ensure_ascii=False,
                )
            )

    @abstractmethod
    async def evaluate(
        self,
        assistant: AssistantProtocol,
        sample: ContextualMCSample,
    ) -> AssistantResult:
        """Evaluate one contextual multiple-choice sample.

        :param assistant: Active assistant episode.
        :param sample: Multiple-choice sample.
        :return: Assistant result for the sample.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dataset(self) -> MappingDataset[ContextualMCSample]:
        """Load the contextual multiple-choice dataset.

        :return: Contextual multiple-choice dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the response evaluator.

        :return: Configured evaluator.
        """
        raise NotImplementedError
