import asyncio
from abc import abstractmethod

from flexrag.common import (
    Context,
    RetrievedContext,
    SimpleProgressLogger,
    configure,
)
from flexrag.common.serialization import json_dumps
from flexrag.datasets.benchmarks import RetrievalDatasetBase
from flexrag.metrics import Evaluator
from flexrag.retrievers import RetrieverProtocol

from .task_base import TaskBase, TaskBaseConfig


@configure
class RetrievalTaskConfig(TaskBaseConfig):
    """Configuration for retrieval evaluation.

    :param batch_size: Number of evaluation queries submitted per search call.
    :param reinit_retriever: Whether to temporarily replace retriever contents
        with the benchmark corpus.
    """

    batch_size: int = 32
    reinit_retriever: bool = False


class RetrievalTask(TaskBase):
    """Base class for batched retrieval evaluation tasks."""

    config: RetrievalTaskConfig
    _logger_name = "task.retrieval"

    def __init__(
        self,
        config: RetrievalTaskConfig,
        *,
        retriever: RetrieverProtocol,
    ) -> None:
        """Load the retrieval dataset and evaluator.

        :param config: Retrieval task configuration.
        :param retriever: Retriever to evaluate.
        :raises ValueError: If ``batch_size`` is not positive.
        """
        if config.batch_size <= 0:
            raise ValueError("RetrievalTaskConfig.batch_size must be positive")
        super().__init__(config)
        self._retriever = retriever
        self.testset = self.load_dataset()
        self.evaluator = self.load_evaluator()

    async def run(self) -> None:
        """Evaluate a retriever with sequential batches of queries.

        When ``reinit_retriever`` is enabled, the benchmark corpus temporarily
        replaces the retriever contents and is cleared even if evaluation
        fails. Existing contents are not restored.

        :raises ValueError: If reinitialization is requested without a corpus.
        """
        retriever = self._retriever
        corpus = self.testset.corpus if self.config.reinit_retriever else None
        if self.config.reinit_retriever:
            if corpus is None:
                raise ValueError(
                    "Dataset corpus is not available. "
                    "Set the dataset to load its corpus before reinitializing the retriever."
                )

        try:
            if corpus is not None:
                if await retriever.async_count() > 0:
                    self.logger.warning(
                        "Retriever is not empty. "
                        "It will be reinitialized for the retrieval task."
                    )
                    await retriever.async_clear()
                await retriever.async_add_contexts(corpus)

            items = list(self.testset)
            questions: list[str] = []
            goldens: list[list[Context]] = []
            retrieved: list[list[RetrievedContext]] = []
            evaluation_qrels: list[dict[str, float]] = []
            self.query_ids: list[str] = []
            self.qrels: dict[str, dict[str, float]] = {}
            p_logger = SimpleProgressLogger(
                self.logger,
                interval=self.config.log_interval,
                total=len(items),
            )

            with open(self.details_path, "w", encoding="utf-8") as f:
                for start in range(0, len(items), self.config.batch_size):
                    batch = items[start : start + self.config.batch_size]
                    batch_results = await retriever.async_search(
                        [item.question for item in batch]
                    )
                    for offset, (item, ctxs) in enumerate(
                        zip(batch, batch_results, strict=True)
                    ):
                        qid = item.question_id or str(start + offset)
                        self.query_ids.append(qid)
                        sample_qrels = dict(item.qrels)
                        self.qrels[qid] = sample_qrels
                        evaluation_qrels.append(sample_qrels)
                        questions.append(item.question)
                        goldens.append(item.contexts or [])
                        retrieved.append(ctxs)
                        f.write(
                            json_dumps(
                                {
                                    "question": item.question,
                                    "golden_contexts": item.contexts,
                                    "qrels": item.qrels,
                                    "metadata": item.metadata,
                                    "contexts": ctxs,
                                },
                                ensure_ascii=False,
                            )
                        )
                        f.write("\n")
                        p_logger.update(desc="Searching")

            eval_score, eval_score_detail = await asyncio.to_thread(
                self.evaluator.evaluate,
                questions=questions,
                retrieved_contexts=retrieved,
                golden_contexts=goldens,
                qrels=evaluation_qrels,
                log=True,
            )

            with open(self.eval_score_path, "w", encoding="utf-8") as f:
                f.write(
                    json_dumps(
                        {
                            "eval_scores": eval_score,
                            "eval_details": eval_score_detail,
                        },
                        indent=4,
                        ensure_ascii=False,
                    )
                )
        finally:
            if self.config.reinit_retriever:
                await retriever.async_clear()

    @abstractmethod
    def load_dataset(self) -> RetrievalDatasetBase:
        """Load the dataset for the Retrieval task.

        :return: The dataset for the Retrieval task.
        :rtype: RetrievalDatasetBase
        """
        return

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the evaluator for the Retrieval task.

        :return: The evaluator for the Retrieval task.
        :rtype: Evaluator
        """
        return
