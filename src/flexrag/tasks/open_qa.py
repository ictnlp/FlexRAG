import logging
import os
import re
from abc import abstractmethod
from collections import Counter
from dataclasses import field
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
    BrowseCompZHDataset,
    BrowseCompZHDatasetConfig,
    SimpleQADataset,
    SimpleQADatasetConfig,
    UDAQADataset,
    UDAQADatasetConfig,
)
from flexrag.datasets.core import MappingDataset, QASample
from flexrag.metrics import (
    F1,
    Evaluator,
    ExactMatch,
    ExactMatchConfig,
    F1Config,
    Rouge,
    RougeConfig,
)
from flexrag.models.generators import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.processors.text_processors import AnswerSimplifier

from .task_base import TASKS, TaskBase


@configure
class OpenQATaskConfig:
    """Configuration for Open Domain QA Task.

    :param log_interval: The interval for logging progress during evaluation.
        Default is 10.
    :type log_interval: int
    :param output_path: The path to save the evaluation results and logs.
        If not specified results and logs will not be saved. Default is None.
    :type output_path: Optional[str]
    """

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
        metadatas: list[dict] = []
        responses: list[str] = []
        contexts: list[list[RetrievedContext]] = []
        p_logger = SimpleProgressLogger(
            self.logger, interval=self.config.log_interval, total=len(self.testset)
        )
        with open(self.details_path, "w", encoding="utf-8") as f:
            for item in self.testset:
                questions.append(item.question)
                golden_answers.append(item.answers)
                metadatas.append(item.meta_data or {})
                response = self.evaluate(assistant=assistant, sample=item)
                responses.append(response.response.text_content or "")
                contexts.append(response.contexts or [])
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
            metadatas=metadatas,
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


class _BrowseCompMetric:
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
class BrowseCompTaskConfig(OpenQATaskConfig, BrowseCompDatasetConfig):
    """Configuration for BrowseComp Task.

    :param llm_judger: The configuration for the LLM judger used in evaluation.
        If not specified, the LLM judger will not be used and the evaluation will only
        include traditional metrics like F1 and Exact Match. Default is None.
    :type llm_judger: GeneratorConfig
    """

    llm_judger: GeneratorConfig = field(default_factory=GeneratorConfig)


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
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.config.llm_judger.generator_type is not None:
            metrics["llm_judger"] = _BrowseCompMetric(self.config.llm_judger)
            self.logger.info("LLM judger is enabled for evaluation.")
        return Evaluator(metrics)

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        prompt = self.template.format(Question=sample.question)
        response = assistant.answer([{"role": "user", "content": prompt}])
        return response


class _BrowseCompZHMetric:
    """The evaluation metric for BrowseComp-ZH Task."""

    template = (
        Path(__file__).parent / "task_prompts" / "browsecomp_zh_metric_prompt.txt"
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
        correct_count = 0
        calibration_bins = [
            {"samples": 0, "correct": 0, "conf_sum": 0.0} for _ in range(5)
        ]
        for output in outputs:
            text = output[0].text_content or ""
            correct_match = re.search(r"correct\s*:\s*(yes|no)", text, re.IGNORECASE)
            confidence_match = re.search(
                r"confidence\s*:\s*([0-9]{1,3})\s*%?", text, re.IGNORECASE
            )
            extracted_answer_match = re.search(
                r"extracted_final_answer\s*:\s*(.*?)\n",
                text,
                re.IGNORECASE | re.DOTALL,
            )

            is_correct = (
                correct_match.group(1).lower() if correct_match is not None else ""
            )
            confidence = int(confidence_match.group(1)) if confidence_match else 100
            confidence = min(max(confidence, 0), 100)
            if is_correct == "yes":
                correct_count += 1

            bin_idx = min(confidence // 20, len(calibration_bins) - 1)
            calibration_bins[bin_idx]["samples"] += 1
            calibration_bins[bin_idx]["conf_sum"] += confidence
            if is_correct == "yes":
                calibration_bins[bin_idx]["correct"] += 1

            details.append(
                {
                    "correct": is_correct,
                    "confidence": confidence,
                    "extracted_final_answer": (
                        extracted_answer_match.group(1).strip()
                        if extracted_answer_match is not None
                        else ""
                    ),
                }
            )

        calibration_error = 0.0
        total = len(questions)
        for stats in calibration_bins:
            if stats["samples"] == 0:
                continue
            accuracy = stats["correct"] / stats["samples"]
            avg_conf = stats["conf_sum"] / stats["samples"] / 100.0
            calibration_error += (stats["samples"] / total) * abs(accuracy - avg_conf)

        return {
            "accuracy": correct_count / total,
            "calibration_error": calibration_error * 100.0,
        }, {"details": details, "calibration_bins": calibration_bins}


@configure
class BrowseCompZHTaskConfig(OpenQATaskConfig, BrowseCompZHDatasetConfig):
    """Configuration for BrowseComp-ZH Task.

    :param llm_judger: The configuration for the LLM judger used in evaluation.
        If not specified, calibration-style evaluation will be disabled.
    :type llm_judger: GeneratorConfig
    """

    llm_judger: GeneratorConfig = field(default_factory=GeneratorConfig)


@TASKS("browsecomp_zh", config_class=BrowseCompZHTaskConfig)
class BrowseCompZHTask(OpenQATask):
    """The BrowseComp-ZH Task for open domain question answering."""

    system_prompt = "你是一个有帮助的中文助手。"
    template = (
        "{Question}\n\n请严格按照以下格式作答：\nExplanation: {{请给出推理说明}}\n"
        "Exact Answer: {{请给出简洁明确的最终答案}}\n"
        "Confidence: {{请给出 0% 到 100% 之间的置信度}}"
    )

    def load_dataset(self) -> MappingDataset[QASample]:
        return BrowseCompZHDataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.config.llm_judger.generator_type is not None:
            metrics["llm_judger"] = _BrowseCompZHMetric(self.config.llm_judger)
            self.logger.info("LLM judger is enabled for evaluation.")
        return Evaluator(metrics)

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        prompt = self.template.format(Question=sample.question)
        return assistant.answer(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )


class _SimpleQAMetric:
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
class SimpleQATaskConfig(OpenQATaskConfig, SimpleQADatasetConfig):
    """Configuration for SimpleQA Task.

    :param llm_judger: The configuration for the LLM judger used in evaluation.
        If not specified, the LLM judger will not be used and the evaluation will only
        include traditional metrics like F1 and Exact Match. Default is None.
    :type llm_judger: GeneratorConfig
    """

    llm_judger: GeneratorConfig = field(default_factory=GeneratorConfig)


@TASKS("simple_qa", config_class=SimpleQATaskConfig)
class SimpleQATask(OpenQATask):
    """The SimpleQA Task for open domain question answering."""

    def load_dataset(self) -> MappingDataset[QASample]:
        return SimpleQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        metrics = {
            "f1": F1(F1Config()),
            "exact_match": ExactMatch(ExactMatchConfig()),
            "rouge": Rouge(RougeConfig()),
        }
        if self.config.llm_judger.generator_type is not None:
            self.logger.info("LLM judger is enabled for evaluation.")
            metrics["llm_judger"] = _SimpleQAMetric(self.config.llm_judger)
        return Evaluator(metrics)

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        response = assistant.answer([{"role": "user", "content": sample.question}])
        return response


class _UDAQAOfficialMetric:
    """Subset-aware official metric wrapper for UDA-QA."""

    def __init__(self, subset: str):
        self.subset = subset
        self.scale_factors = {
            "": 1.0,
            "thousand": 1_000.0,
            "million": 1_000_000.0,
            "billion": 1_000_000_000.0,
            "percent": 1.0,
        }
        self.number_pattern = re.compile(r"\d[\d,]*(?:\.\d+)?")
        self.scale_pattern = re.compile(
            r"\b(?:thousand|thousands|million|millions|billion|billions|percent|bn|mn|k)\b|%"
        )
        self.simplifier = AnswerSimplifier()
        return

    def _token_f1(
        self, golds_tokens: list[list[str]], response_tokens: list[str]
    ) -> float:
        best_f1 = 0.0
        response_counter = Counter(response_tokens)
        for gold_tokens in golds_tokens:
            gold_counter = Counter(gold_tokens)
            common = sum((gold_counter & response_counter).values())
            if common == 0:
                continue
            precision = common / max(len(response_tokens), 1)
            recall = common / max(len(gold_tokens), 1)
            best_f1 = max(best_f1, (2 * precision * recall) / (precision + recall))
        return best_f1

    def _detect_scale_factor(self, context: str, default_scale: str) -> float:
        lowered = context.lower()
        if "%" in lowered or "percent" in lowered:
            return self.scale_factors["percent"]
        if "billion" in lowered or re.search(r"\bbn\b", lowered):
            return self.scale_factors["billion"]
        if "million" in lowered or re.search(r"\bmn\b", lowered):
            return self.scale_factors["million"]
        if "thousand" in lowered or re.search(r"\bk\b", lowered):
            return self.scale_factors["thousand"]
        return self.scale_factors.get(default_scale.lower(), 1.0)

    def _extract_numeric_values(
        self, text: str, default_scale: str = ""
    ) -> list[float]:
        lowered = str(text).lower()
        values = []
        for match in self.number_pattern.finditer(lowered):
            raw = match.group().replace(",", "")
            value = float(raw)
            prefix = lowered[max(0, match.start() - 6) : match.start()]
            suffix = lowered[match.end() : match.end() + 24]
            if "(" in prefix and ")" in suffix and not prefix.rstrip().endswith("-"):
                value = -value
            if prefix.rstrip().endswith("-"):
                value = -value
            factor = self._detect_scale_factor(f"{prefix} {suffix}", default_scale)
            values.append(value * factor)
        return values

    def _basic_f1_score(self, golds: list[str], response: str) -> float:
        golds_tokens = [self.simplifier(str(gold)).split() for gold in golds]
        response_tokens = self.simplifier(str(response)).split()
        return self._token_f1(golds_tokens, response_tokens)

    def _fin_exact_match(self, golds: list[str], response: str) -> float:
        response_values = self._extract_numeric_values(response)
        if len(response_values) == 1:
            for gold in golds:
                gold_values = self._extract_numeric_values(gold)
                if len(gold_values) != 1:
                    continue
                gold_value = gold_values[0]
                diff = abs(response_values[0] - gold_value)
                tolerance = abs(gold_value) * 0.01
                if gold_value == 0:
                    tolerance = 1e-9
                if diff <= tolerance:
                    return 1.0
        simplified_response = self.simplifier(str(response))
        return float(
            any(self.simplifier(str(gold)) == simplified_response for gold in golds)
        )

    def _tat_tokens(self, text: str, answer_scale: str) -> list[str]:
        values = [
            format(value, ".15g")
            for value in self._extract_numeric_values(text, answer_scale)
        ]
        text_without_numbers = self.number_pattern.sub(" ", str(text))
        text_without_scales = self.scale_pattern.sub(" ", text_without_numbers)
        text_tokens = self.simplifier(text_without_scales).split()
        return text_tokens + values

    def _tat_f1_score(self, golds: list[str], response: str, metadata: dict) -> float:
        answer_type = str(metadata.get("answer_type", ""))
        answer_scale = str(metadata.get("answer_scale", ""))
        response_tokens = self._tat_tokens(response, answer_scale)
        if answer_type == "multi-span":
            golds_tokens = [self._tat_tokens(" ".join(golds), answer_scale)]
        else:
            golds_tokens = [self._tat_tokens(gold, answer_scale) for gold in golds]
        return self._token_f1(golds_tokens, response_tokens)

    def __call__(
        self,
        responses: list[str],
        golden_responses: list[list[str]],
        metadatas: list[dict],
    ):
        scores = []
        for response, golds, metadata in zip(responses, golden_responses, metadatas):
            if self.subset in {"feta", "nq", "paper_text", "paper_tab"}:
                score = self._basic_f1_score(golds, response)
            elif self.subset == "fin":
                score = self._fin_exact_match(golds, response)
            else:
                score = self._tat_f1_score(golds, response, metadata)
            scores.append(score)
        official_score = sum(scores) / len(scores) if scores else 0.0
        return {"official_score": official_score}, {"item_score": scores}


@configure
class UDAQATaskConfig(OpenQATaskConfig, UDAQADatasetConfig):
    """Configuration for UDAQA Task."""


@TASKS("uda_qa", config_class=UDAQATaskConfig)
class UDAQATask(OpenQATask):
    """The UDA-QA Task for file-grounded question answering."""

    instruction = (
        "Read the attached document and answer the question. "
        "Return only the final answer without extra explanation.\n\n"
        "Question: {question}"
    )

    def load_dataset(self) -> MappingDataset[QASample]:
        return UDAQADataset(self.config)

    def load_evaluator(self) -> Evaluator:
        return Evaluator(
            {
                "f1": F1(F1Config()),
                "exact_match": ExactMatch(ExactMatchConfig()),
                "rouge": Rouge(RougeConfig()),
                "official_score": _UDAQAOfficialMetric(self.config.subset),
            }
        )

    def evaluate(self, assistant: AssistantBase, sample: QASample) -> AssistantResponse:
        meta_data = sample.meta_data or {}
        file_path = meta_data["source_file_path"]
        file_format = meta_data["source_file_format"]
        prompt = self.instruction.format(question=sample.question)
        if file_format == "pdf":
            file_block = {"type": "pdf", "file_path": file_path}
        else:
            file_block = {
                "type": "file",
                "file_path": file_path,
                "mime_type": meta_data["source_mime_type"],
                "file_name": meta_data["source_file_name"],
            }
        return assistant.answer(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        file_block,
                    ],
                }
            ]
        )
