import logging
import os
from dataclasses import field
from pathlib import Path
from typing import Optional

from flexrag.database import json_dump
from flexrag.datasets import RetrievalDataset
from flexrag.metrics import Evaluator, EvaluatorConfig
from flexrag.retriever import RetrieverBase
from flexrag.utils import LOGGER_MANAGER, SimpleProgressLogger, configure

from .tasks import TASKS, TaskBase


@configure
class RetrievalTaskConfig:
    """Configuration for Retrieval Task."""

    eval_config: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    log_interval: int = 10
    output_path: Optional[str] = None


@TASKS("retrieval", config_name=RetrievalTaskConfig)
class RetrievalTask(TaskBase):
    """Retrieval Task."""

    config: RetrievalTaskConfig

    def setup(self, retriever: RetrieverBase, dataset: RetrievalDataset):
        """Setup the Retrieval task."""
        self.retriever = retriever
        assert len(self.retriever) == 0, "Retriever is not empty."
        self.testset = dataset
        self.evaluator = Evaluator(self.config.eval_config)

        # prepare output path
        if self.config.output_path is not None:
            if not Path(self.config.output_path).exists():
                Path(self.config.output_path).mkdir(exist_ok=True, parents=True)
            config_path = Path(self.config.output_path, "config.yaml")
            log_path = Path(self.config.output_path, "log.txt")
        else:
            config_path = Path(os.devnull)
            log_path = Path(os.devnull)

        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.retrieval")
        handler = logging.FileHandler(log_path)
        LOGGER_MANAGER.add_handler(handler)
        self.logger.debug(f"Configs:\n{self.config.dumps()}")
        self.config.dump(config_path)
        return

    def run(self):
        """Run the Retrieval task."""
        # prepare output paths
        if self.config.output_path is not None:
            details_path = Path(self.config.output_path, "details.jsonl")
            eval_score_path = Path(self.config.output_path, "eval_score.json")
        else:
            details_path = Path(os.devnull)
            eval_score_path = Path(os.devnull)

        # search and answer questions
        questions = []
        goldens = []
        retrieved = []
        p_logger = SimpleProgressLogger(self.logger, interval=self.config.log_interval)
        with open(details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                questions.append(item.question)
                goldens.append(item.contexts)
                ctxs = self.retriever.search(query=item.question)[0]
                retrieved.append(ctxs)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden_contexts": item.contexts,
                            "metadata": item.meta_data,
                            "contexts": ctxs,
                        },
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

        # Save the evaluation results
        with open(eval_score_path, "w", encoding="utf-8") as f:
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
