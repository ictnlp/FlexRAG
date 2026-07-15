import asyncio
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable

from flexrag.assistants import AssistantProtocol, AssistantResult
from flexrag.common import RetrievedContext, SimpleProgressLogger, configure
from flexrag.common.serialization import json_dumps
from flexrag.datasets.core import MappingDataset, MultiSessionQASample
from flexrag.metrics import Evaluator

from .task_base import TaskBase, TaskBaseConfig


@configure
class MultiSessionQATaskConfig(TaskBaseConfig):
    """Configuration for multi-session QA evaluation.

    :param log_interval: Progress logging interval.
    :param output_path: Optional directory for evaluation outputs.
    """


class MultiSessionQATask(TaskBase):
    """Base class for episode-isolated multi-session QA tasks."""

    config: MultiSessionQATaskConfig
    _logger_name = "task.multi_session_qa"

    def __init__(
        self,
        config: MultiSessionQATaskConfig,
        *,
        assistant_factory: Callable[[], AssistantProtocol],
    ) -> None:
        """Load the dataset and evaluator for a multi-session QA task.

        :param config: Multi-session QA task configuration.
        :param assistant_factory: Factory returning a fresh assistant for each
            context group.
        """
        super().__init__(config)
        self._assistant_factory = assistant_factory
        self.testset = self.load_dataset()
        self.evaluator = self.load_evaluator()

    async def run(self) -> None:
        """Evaluate groups serially and questions within each group concurrently."""
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
            async with self._assistant_factory() as assistant:
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
            retrieved_contexts=contexts,
            metadata=metadatas,
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
