from abc import abstractmethod
from collections import Counter
from dataclasses import field

from flexrag.common import TIME_METER, configure
from flexrag.models.tokenizer import TOKENIZERS, TokenizerConfig
from flexrag.processors.text_processors import AnswerSimplifier

from .metrics_base import METRICS


@configure
class MatchingMetricsConfig:
    """Configuration class for MatchingMetrics.

    :param simplify: Whether to simplify the answer before computing the matching score.
        Defaults to True.
    :type simplify: bool
    """

    simplify: bool = True


class MatchingMetrics:
    name: str

    def __init__(self, cfg: MatchingMetricsConfig) -> None:
        if cfg.simplify:
            self.simplifier = AnswerSimplifier()
        else:
            self.simplifier = None
        return

    @abstractmethod
    def compute_item(self, golds: list[str], response: str) -> float:
        return

    @TIME_METER("metrics.matching_score")
    def __call__(
        self, responses: list[str], golden_responses: list[list[str]]
    ) -> tuple[dict[str, float], dict[str, list[float]]]:
        if self.simplifier is not None:
            responses = [self.simplifier(response) for response in responses]
            golden_responses = [
                [self.simplifier(gold) for gold in golds] for golds in golden_responses
            ]
        matching_list = []
        for golds, response in zip(golden_responses, responses):
            matching_list.append(self.compute_item(golds, response))
        matching_score = sum(matching_list) / len(matching_list)
        return {self.name: matching_score}, {"item_score": matching_list}


ExactMatchConfig = MatchingMetricsConfig
AccuracyConfig = MatchingMetricsConfig


@METRICS("generation_em", config_class=ExactMatchConfig)
class ExactMatch(MatchingMetrics):
    """ExactMatch metric computes if any of the golden responses is exactly the same as the predicted response."""

    name = "generation_em"

    def compute_item(self, golds: list[str], response: str) -> float:
        return float(response in golds)


@METRICS("generation_accuracy", config_class=AccuracyConfig)
class Accuracy(MatchingMetrics):
    """Accuracy metric computes if any of the golden responses is in the predicted response."""

    name = "generation_accuracy"

    def compute_item(self, golds: list[str], response: str) -> float:
        return float(any(gold in response for gold in golds))


def f1_recall_precision(
    golds_tokens: list[list[str]], response_tokens: list[str]
) -> tuple[float, float, float]:
    true_counters = [Counter(gold_tokens) for gold_tokens in golds_tokens]
    pred_counter = Counter(response_tokens)
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for gold in true_counters:
        common = sum((gold & pred_counter).values())
        if common == 0:
            continue
        p = 1.0 * common / sum(pred_counter.values())
        r = 1.0 * common / sum(gold.values())
        f1_ = (2 * p * r) / (p + r)
        precision = max(p, precision)
        recall = max(r, recall)
        f1 = max(f1, f1_)
    return f1, recall, precision


@configure
class F1RecallPrecisionConfig(MatchingMetricsConfig):
    """Configuration class for F1RecallPrecision metric.

    :param tokenizer_config: The tokenizer used for splitting the answer into tokens.
        Defaults to a space tokenizer.
    :type tokenizer_config: TokenizerConfig
    """

    tokenizer_config: TokenizerConfig = field(
        default_factory=lambda: TokenizerConfig(tokenizer_type="space")
    )


F1Config = F1RecallPrecisionConfig
RecallConfig = F1RecallPrecisionConfig
PrecisionConfig = F1RecallPrecisionConfig


@METRICS("generation_f1", config_class=F1Config)
class F1(MatchingMetrics):
    """F1 metric computes the F1 score of the predicted response against the golden responses."""

    name = "generation_f1"

    def __init__(self, cfg: F1Config) -> None:
        super().__init__(cfg)
        self.tokenizer = TOKENIZERS.load(cfg.tokenizer_config)
        return

    def compute_item(self, golds: list[str], response: str) -> float:
        golds_tokens: list[list[str]] = [
            self.tokenizer.tokenize(gold) for gold in golds
        ]
        response_tokens: list[str] = self.tokenizer.tokenize(response)
        return f1_recall_precision(golds_tokens, response_tokens)[0]


@METRICS("generation_recall", config_class=RecallConfig)
class Recall(MatchingMetrics):
    """Recall metric computes the recall of the predicted response against the golden responses."""

    name = "generation_recall"

    def __init__(self, cfg: RecallConfig) -> None:
        super().__init__(cfg)
        self.tokenizer = TOKENIZERS.load(cfg.tokenizer_config)
        return

    def compute_item(self, golds: list[str], response: str) -> float:
        golds_tokens: list[list[str]] = [
            self.tokenizer.tokenize(gold) for gold in golds
        ]
        response_tokens: list[str] = self.tokenizer.tokenize(response)
        return f1_recall_precision(golds_tokens, response_tokens)[1]


@METRICS("generation_precision", config_class=PrecisionConfig)
class Precision(MatchingMetrics):
    """Precision metric computes the precision of the predicted response against the golden responses."""

    name = "generation_precision"

    def __init__(self, cfg: PrecisionConfig) -> None:
        super().__init__(cfg)
        self.tokenizer = TOKENIZERS.load(cfg.tokenizer_config)
        return

    def compute_item(self, golds: list[str], response: str) -> float:
        golds_tokens: list[list[str]] = [
            self.tokenizer.tokenize(gold) for gold in golds
        ]
        response_tokens: list[str] = self.tokenizer.tokenize(response)
        return f1_recall_precision(golds_tokens, response_tokens)[2]
