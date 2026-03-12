import logging
import os
from abc import abstractmethod
from dataclasses import field
from pathlib import Path
from typing import Optional

from flexrag.common import LOGGER_MANAGER, Context, SimpleProgressLogger, configure
from flexrag.common.database import json_dump
from flexrag.datasets.benchmarks import (
    MSMARCODataset,
    MSMARCODatasetConfig,
    MTEBDataset,
    MTEBDatasetConfig,
    MultiLongDocRetrievalDataset,
    MultiLongDocRetrievalDatasetConfig,
    RetrievalDatasetBase,
)
from flexrag.metrics import (
    Evaluator,
    RetrievalMAP,
    RetrievalMAPConfig,
    RetrievalMRR,
    RetrievalNDCG,
    RetrievalNDCGConfig,
    RetrievalRecall,
    RetrievalRecallConfig,
)
from flexrag.retrievers import RetrieverBase

from .task_base import TASKS, TaskBase


@configure
class RetrievalTaskConfig:
    """Configuration for Retrieval Task."""

    log_interval: int = 10
    output_path: Optional[str] = None
    reinit_retriever: bool = False


class RetrievalTask(TaskBase):
    """Retrieval Task."""

    config: RetrievalTaskConfig

    def setup(self):
        """Setup the Retrieval task."""

        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.retrieval")
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

    def run(self, retriever: RetrieverBase):
        """Run the Retrieval task."""
        # initial check
        if self.config.reinit_retriever:
            if len(retriever) > 0:
                self.logger.warning(
                    "Retriever is not empty. "
                    "It will be reinitialized for the retrieval task."
                )
                retriever.clear()
            retriever.add_passages(self.testset.contexts.values())

        # search and answer questions
        questions: list[str] = []
        goldens: list[list[Context]] = []
        retrieved: list[list[Context]] = []
        p_logger = SimpleProgressLogger(self.logger, interval=self.config.log_interval)
        with open(self.details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                questions.append(item.question)
                goldens.append(item.contexts)
                ctxs = retriever.search(query=item.question)[0]
                retrieved.append(ctxs)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden_contexts": item.contexts,
                            "metadata": item.meta_data,
                            "contexts": ctxs,
                        },
                        to_bytes=False,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                p_logger.update(desc="Searching")

        # Evaluate the results
        eval_score, eval_score_detail = self.evaluator.evaluate(
            questions=questions,
            retrieved_contexts=retrieved,
            golden_contexts=goldens,
            log=True,
        )

        # clean up retriever if needed
        if self.config.reinit_retriever:
            retriever.clear()

        # Save the evaluation results
        with open(self.eval_score_path, "w", encoding="utf-8") as f:
            f.write(
                json_dump(
                    {
                        "eval_scores": eval_score,
                        "eval_details": eval_score_detail,
                    },
                    to_bytes=False,
                    indent=4,
                    ensure_ascii=False,
                )
            )
        return

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


@configure
class MTEBRetrievalTaskConfig(RetrievalTaskConfig, MTEBDatasetConfig):
    """Configuration for MTEB Retrieval Task."""


@TASKS("mteb")
class MTEBRetrievalTask(RetrievalTask):
    """MTEB Retrieval Task."""

    def load_dataset(self) -> MTEBDataset:
        """Load the MTEB dataset for the Retrieval task.

        :return: The MTEB dataset for the Retrieval task.
        :rtype: MTEBDataset
        """
        return MTEBDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)


@configure
class MLDRRetrievalTaskConfig(RetrievalTaskConfig, MultiLongDocRetrievalDatasetConfig):
    """Configuration for MLDR Retrieval Task."""


@TASKS("mldr", config_class=MLDRRetrievalTaskConfig)
class MLDRRetrievalTask(RetrievalTask):
    """MLDR Retrieval Task."""

    def load_dataset(self) -> MultiLongDocRetrievalDataset:
        """Load the MLDR dataset for the Retrieval task.

        :return: The MLDR dataset for the Retrieval task.
        :rtype: MultiLongDocRetrievalDataset
        """
        return MultiLongDocRetrievalDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)


@configure
class MSMARCORetrievalTaskConfig(RetrievalTaskConfig, MSMARCODatasetConfig):
    """Configuration for MSMARCO Retrieval Task."""


@TASKS("ms_marco", config_class=MSMARCORetrievalTaskConfig)
class MSMARCORetrievalTask(RetrievalTask):
    """MSMARCO Retrieval Task."""

    def load_dataset(self) -> MSMARCODataset:
        """Load the MSMARCO dataset for the Retrieval task.

        :return: The MSMARCO dataset for the Retrieval task.
        :rtype: MSMARCODataset
        """
        return MSMARCODataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "ndcg": RetrievalNDCG(RetrievalNDCGConfig(k_values=[1, 3, 5, 10])),
            "recall": RetrievalRecall(RetrievalRecallConfig(k_values=[1, 3, 5, 10])),
            "map": RetrievalMAP(RetrievalMAPConfig(k_values=[1, 3, 5, 10])),
            "mrr": RetrievalMRR(),
        }
        return Evaluator(metrics)
