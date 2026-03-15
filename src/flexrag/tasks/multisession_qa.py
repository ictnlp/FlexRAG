import json
import logging
import os
from abc import abstractmethod
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import Optional

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import (
    LOGGER_MANAGER,
    RetrievedContext,
    SimpleProgressLogger,
    configure,
)
from flexrag.common.database import json_dump
from flexrag.datasets.benchmarks import (
    ConvoMemDataset,
    ConvoMemDatasetConfig,
    LoCoMoDataset,
    LoCoMoDatasetConfig,
)
from flexrag.datasets.core import MappingDataset, MultiSessionQASample
from flexrag.metrics import (
    F1,
    Accuracy,
    AccuracyConfig,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)
from flexrag.models.generators import GENERATORS, GenerationConfig, GeneratorConfig

from .task_base import TASKS, TaskBase


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
                    contexts.append(response.contexts)
                    metadatas.append(item.meta_data)
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


@configure
class LoCoMoTaskConfig(MultiSessionQATaskConfig, LoCoMoDatasetConfig):
    """Configuration for LoCoMo Task."""


@TASKS("locomo", config_class=LoCoMoTaskConfig)
class LoCoMoTask(MultiSessionQATask):
    """LoCoMo Task."""

    def load_dataset(self) -> LoCoMoDataset:
        return LoCoMoDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "em": ExactMatch(ExactMatchConfig()),
            "f1": F1(F1Config()),
            "rouge": Rouge(RougeConfig()),
            "accuracy": Accuracy(AccuracyConfig()),
        }
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: MultiSessionQASample
    ) -> AssistantResponse:
        return assistant.answer([{"role": "user", "content": sample.question}])


class _ConvoMemMetric:
    templates = json.load(
        open(
            Path(__file__).parent / "task_prompts" / "convomem_metric_prompt.json",
            mode="r",
            encoding="utf-8",
        )
    )

    def __init__(self, cfg: GeneratorConfig, subset: str):
        self.generator = GENERATORS.load(cfg)
        self.gen_cfg = GenerationConfig(do_sample=False)
        self.subset = subset
        self.default = 0
        assert self.generator is not None, "Generator is not loaded"
        return

    def __call__(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        metadata: list[dict],
    ):
        match self.subset:
            case "abstention_evidence":
                template = self.templates["abstention"]
                params = {
                    "question": questions,
                    "response": responses,
                }
            case s if s in {
                "assistant_facts_evidence",
                "changing_evidence",
                "implicit_connection_evidence",
            }:
                template = self.templates["factual"]
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                }
            case "preference_evidence":
                template = self.templates["rubric"]
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                }
            case "user_evidence":
                template = self.templates["user_facts"]
                evidences = []
                for meta in metadata:
                    evidence = ""
                    for n, conv in enumerate(meta["message_evidences"]):
                        evidence += f"Message Evidence {n}: {conv['text']}\n"
                    evidences.append(evidence.rstrip())
                params = {
                    "question": questions,
                    "response": responses,
                    "golden_answer": [g[0] for g in golden_responses],
                    "evidence": evidences,
                }
            case _:
                raise ValueError(f"Unsupported subset: {self.subset}")
        prompts = []
        for row in zip(*params.values()):
            keys = {k: v for k, v in zip(params.keys(), row)}
            prompt = template.format(**keys)
            prompts.append([{"role": "user", "content": prompt}])
        outputs = self.generator.chat(prompts, self.gen_cfg)

        # compute scores
        scores = []
        for output in outputs:
            text = output[0].text_content.strip().lower() or ""
            if "right" in text:
                scores.append(1)
            elif "wrong" in text:
                scores.append(0)
            else:
                scores.append(self.default)
        accuracy = sum(scores) / len(scores) if len(scores) > 0 else 0.0
        return {"accuracy": accuracy}, {"details": scores}


@configure
class ConvoMemTaskConfig(MultiSessionQATaskConfig, ConvoMemDatasetConfig):
    """Configuration for ConvoMem Task.

    :param llm_judger: The configuration for the LLM judger used in evaluation.
        If not specified, the LLM judger will not be used and the evaluation will only
        include traditional metrics like F1 and Exact Match. Default is None.
    :type llm_judger: GeneratorConfig
    """

    llm_judger: GeneratorConfig = field(default_factory=GeneratorConfig)


@TASKS("convomem", config_class=ConvoMemTaskConfig)
class ConvoMemTask(MultiSessionQATask):
    """ConvoMem Task."""

    def load_dataset(self) -> ConvoMemDataset:
        return ConvoMemDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.config.llm_judger.generator_type is not None:
            self.logger.info("LLM judger is enabled for evaluation.")
            metrics["llm_judger"] = _ConvoMemMetric(
                self.config.llm_judger, self.config.subset
            )
        return Evaluator(metrics)

    def evaluate(
        self, assistant: AssistantBase, sample: MultiSessionQASample
    ) -> AssistantResponse:
        return assistant.answer([{"role": "user", "content": sample.question}])
