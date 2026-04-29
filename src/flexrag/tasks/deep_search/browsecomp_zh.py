import re
from dataclasses import field

from flexrag.assistants import AssistantBase, AssistantResponse
from flexrag.common import ChatMessages, configure
from flexrag.datasets.benchmarks import BrowseCompZHDataset, BrowseCompZHDatasetConfig
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

from ..open_qa_base import OpenQATask, OpenQATaskConfig
from ..task_base import TASKS

_METRIC_TEMPLATE = """
请根据给定的[question]、模型的[response]以及唯一且明确的[correct_answer]，判断回答是否正确，并抽取最终答案与置信度。

[question]: {question}

[response]: {response}

请严格按以下格式输出：

extracted_final_answer: 从[response]中抽取出的最终明确答案；如果没有可抽取的最终答案，填写'None'。

[correct_answer]: {correct_answer}

reasoning: 仅基于[correct_answer]判断extracted_final_answer是否正确，说明两者是否存在实质差异。不要补充题目背景，不要重新解题，也不要讨论任何不同于[correct_answer]的答案。

correct: 如果extracted_final_answer与[correct_answer]一致，或者在数值题中处于一个很小的误差范围内，则填写'yes'；否则填写'no'。

confidence: 从[response]中抽取的置信度，范围为0%到100%；如果[response]中没有置信度，填写100。
""".strip()


class _BrowseCompZHMetric:
    """The evaluation metric for BrowseComp-ZH Task."""

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
            prompt = _METRIC_TEMPLATE.format(
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
