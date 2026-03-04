import logging
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import (
    LOGGER_MANAGER,
    ChatMessages,
    RetrievedContext,
    SimpleProgressLogger,
    configure,
)
from flexrag.common.database import json_dump
from flexrag.datasets.benchmarks import (
    BrowseCompDataset,
    BrowseCompDatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
)
from flexrag.datasets.core import MappingDataset, QASample
from flexrag.metrics import Evaluator
from flexrag.models.generators import GENERATORS, GenerationConfig, GeneratorConfig

from .task_base import TASKS, TaskBase


@configure
class OpenQATaskConfig:
    """Configuration for Open Domain QA Task."""

    log_interval: int = 10
    output_path: Optional[str] = None


class OpenQATask(TaskBase):
    """Base class for all Open Domain QA Tasks."""

    config: OpenQATaskConfig

    def setup(self):
        """Setup the Open Domain QA task."""
        # setup logger
        self.logger = LOGGER_MANAGER.get_logger("task.open_qa")
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
        """Run the Open Domain QA task."""
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
            retrieved_contexts=contexts,
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
    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        """
        Evaluate a single data sample.

        :param assistant: The assistant to evaluate.
        :type assistant: AssistantBase
        :param sample: A single data sample to be evaluated.
        :type sample: QASample
        :return: The response from the assistant.
        :rtype: AssistantResponse
        """
        return

    @abstractmethod
    def load_dataset(self) -> MappingDataset[QASample]:
        """Load the dataset for the task.

        :return: The dataset for the task.
        :rtype: MappingDataset[QASample]
        """
        return

    @abstractmethod
    def load_evaluator(self) -> Evaluator:
        """Load the evaluator for the task.

        :return: The evaluator for the task.
        :rtype: Evaluator
        """
        return


class BrowseCompMetric:
    """The evaluation metric for BrowseComp Task."""

    template = (
        Path(__file__).parent / "task_prompts" / "browsecomp_metric_prompt.txt"
    ).read_text(encoding="utf-8")

    def __init__(self, cfg: GeneratorConfig):
        self.generator = GENERATORS.load(cfg)
        self.gen_cfg = GenerationConfig(do_sample=False)
        assert self.generator is not None, "Generator is not loaded."
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        **kwargs,
    ):
        prompts = []
        for question, response, golden_response in zip(
            questions, responses, golden_responses
        ):
            prompt = self.template.format(
                question=question,
                response=response,
                correct_answer=golden_response[0],
            )
            prompts.append(
                ChatMessages.from_list([{"role": "user", "content": prompt}])
            )
        outputs = self.generator.chat(prompts, self.gen_cfg)
        details = []
        for output in outputs:
            text = output[0].text_content or ""
            match = re.search(r"correct: (yes|no)", text)
            # Default to "no" if no match
            details.append(match.group(0) if match else "no")
        correct_count = sum(1 for detail in details if detail == "correct: yes")
        accuracy = correct_count / len(questions)
        return {"accuracy": accuracy}, {"details": details}


@configure
class BrowseCompTaskConfig(OpenQATaskConfig, BrowseCompDatasetConfig, GeneratorConfig):
    """Configuration for BrowseComp Task."""


@TASKS("browsecomp", config_class=BrowseCompTaskConfig)
class BrowseCompTask(OpenQATask):
    """The BrowseComp Task for open domain question answering."""

    template = (
        "{Question}\n\nYour response should be in the following format:\nExplanation:"
        " {{your explanation for your final answer}}\nExact Answer: {{your succinct,"
        " final answer}}\nConfidence: {{your confidence score between 0% and 100% for"
        " your answer}}"
    )

    def load_dataset(self) -> MappingDataset[QASample]:
        return BrowseCompDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metric = BrowseCompMetric(self.config)
        return Evaluator({"browsecomp_accuracy": metric})

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        prompt = self.template.format(Question=sample.question)
        response = assistant.answer([{"role": "user", "content": prompt}])
        return response


class SimpleQAMetric:
    """The evaluation metric for SimpleQA Task."""

    template = (
        Path(__file__).parent / "task_prompts" / "simple_qa_metric_prompt.txt"
    ).read_text(encoding="utf-8")

    def __init__(self, cfg: GeneratorConfig):
        self.generator = GENERATORS.load(cfg)
        self.gen_cfg = GenerationConfig(do_sample=False)
        assert self.generator is not None, "Generator is not loaded."
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        **kwargs,
    ):
        prompts = []
        for question, response, golden_response in zip(
            questions, responses, golden_responses
        ):
            prompt = self.template.format(
                question=question,
                predicted_answer=response,
                target=golden_response[0],
            )
            prompts.append(
                ChatMessages.from_list([{"role": "user", "content": prompt}])
            )

        outputs = self.generator.chat(prompts, self.gen_cfg)

        # compute accuracy
        grades = []
        for output in outputs:
            text = output[0].text_content or ""
            match = re.search(r"(A|B|C)", text)
            grade = match.group(0) if match else "C"  # default to NOT_ATTEMPTED
            grades.append(grade)

        correct_count = sum(1 for grade in grades if grade == "A")
        not_attempted_count = sum(1 for grade in grades if grade == "C")
        accuracy = correct_count / len(questions)
        not_attempted_ratio = not_attempted_count / len(questions)
        return {
            "shortform_correctness": accuracy,
            "not_attempted_ratio": not_attempted_ratio,
        }, {"detailed_grades": grades}


@configure
class SimpleQATaskConfig(OpenQATaskConfig, SimpleQADatasetConfig, GeneratorConfig):
    """Configuration for SimpleQA Task."""


@TASKS("simple_qa", config_class=SimpleQATaskConfig)
class SimpleQATask(OpenQATask):
    """The SimpleQA Task for open domain question answering."""

    def load_dataset(self) -> MappingDataset[QASample]:
        return SimpleQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metric = SimpleQAMetric(self.config)
        return Evaluator({"simple_qa_accuracy": metric})

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        response = assistant.answer([{"role": "user", "content": sample.question}])
        return response
