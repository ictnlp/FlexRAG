from dataclasses import dataclass, field

from flexrag.utils import TIME_METER, Choices

from .xfinder_utils import Evaluator
from .metrics_base import METRICS, MetricsBase


@dataclass
class xFinderConfig:
    model_type: Choices(["qwen", "llama"]) = "qwen"  # type: ignore
    model_path: str = "IAAR-Shanghai/xFinder-qwen1505"
    answer_type: Choices(  # type: ignore
        ["math", "short_text", "categorical_label", "alphabet_option"]
    ) = "short_text"
    temperature: float = 0.7
    max_tokens: int = 100
    device_id: list[int] = field(default_factory=list)


@METRICS("generation_xfinder")
class xFinder(MetricsBase):
    def __init__(self, config: xFinderConfig):
        if config.model_type == "qwen":
            model_name = "xFinder-qwen1505"
        else:
            model_name = "xFinder-llama38it"
        self.evaluator = Evaluator(
            model_name=model_name,
            model_path_or_url=config.model_path,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            device_id=config.device_id,
        )
        self.answer_type = config.answer_type
        return

    @TIME_METER("metrics.xfinder_score")
    def compute(
        self,
        questions: list[str],
        responses: list[str],
        golden_responses: list[list[str]],
        choices: list[list[str]],
        **kwargs
    ) -> tuple[float, dict[str, list[float]]]:
        results = []
        for question, response, goldens, choice in zip(
            questions, responses, golden_responses, choices
        ):
            self.evaluator.evaluate_single_item(
                question=question,
                llm_output=response,
                answer_type=self.answer_type,
                correct_answer=goldens[0],
                answer_range=",".join(choice),
            )

        correct_count = sum(result[-1] for result in results)
        accuracy = correct_count / max(1, len(results)) if results else 0
        return accuracy
