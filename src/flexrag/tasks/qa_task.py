import logging
import os
from dataclasses import field
from pathlib import Path
from typing import Optional

from flexrag.assistants import ASSISTANTS, AssistantConfig
from flexrag.common import LOGGER_MANAGER, SimpleProgressLogger, configure
from flexrag.common.database import json_dump
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.datasets import QA_DATASETS, QADatasetConfig
from flexrag.metrics import Evaluator, EvaluatorConfig

from .tasks import TASKS, TaskBase

PREDEFINED_PROMPTS = {
    "shortform": ChatMessages.from_json(
        Path(__file__).parent / "task_prompts" / "shortform_qa.json"
    ),
    "longform": ChatMessages.from_json(
        Path(__file__).parent / "task_prompts" / "longform_qa.json"
    ),
}


@configure
class QATaskConfig(AssistantConfig, QADatasetConfig, EvaluatorConfig):
    """Configuration for Knowledge Intensive QA Task."""

    log_interval: int = 10
    output_path: Optional[str] = None


@TASKS("qa", config_class=QATaskConfig)
class QATask(TaskBase):
    """Knowledge Intensive QA Task."""

    config: QATaskConfig

    def setup(self):
        """Setup the Knowledge Intensive QA task."""
        self.assistant = ASSISTANTS.load(self.config)
        self.testset = QA_DATASETS.load(self.config)
        self.evaluator = Evaluator(self.config)

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
        self.logger = LOGGER_MANAGER.get_logger("task.qa")
        handler = logging.FileHandler(log_path)
        LOGGER_MANAGER.add_handler(handler)
        self.logger.debug(f"Configs:\n{self.config.dumps()}")
        self.config.dump(config_path)
        return

    def make_prompt(self, question: str) -> ChatMessages:
        if self.testset.form == "short":
            prompt = PREDEFINED_PROMPTS["shortform"].copy()
        else:
            prompt = PREDEFINED_PROMPTS["longform"].copy()
        prompt.append(ChatTurn(role="user", content=question))
        return prompt

    def run(self):
        """Run the Knowledge Intensive QA task."""
        # prepare output paths
        if self.config.output_path is not None:
            details_path = Path(self.config.output_path, "details.jsonl")
            eval_score_path = Path(self.config.output_path, "eval_score.json")
        else:
            details_path = Path(os.devnull)
            eval_score_path = Path(os.devnull)

        # search and answer questions
        questions = []
        golden_answers = []
        golden_contexts = []
        responses = []
        contexts: list[list[RetrievedContext]] = []
        p_logger = SimpleProgressLogger(self.logger, interval=self.config.log_interval)
        with open(details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                prompt = self.make_prompt(item.question)
                questions.append(item.question)
                golden_answers.append(item.golden_answers)
                golden_contexts.append(item.golden_contexts)
                response = self.assistant.answer(messages=prompt)
                responses.append(response.response)
                contexts.append(response.contexts)
                f.write(
                    json_dump(
                        {
                            "question": item.question,
                            "golden": item.golden_answers,
                            "golden_contexts": item.golden_contexts,
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
            retrieved_contexts=contexts,
            golden_contexts=golden_contexts,
            log=True,
        )

        # Save the evaluation results
        with open(eval_score_path, "w", encoding="utf-8") as f:
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
