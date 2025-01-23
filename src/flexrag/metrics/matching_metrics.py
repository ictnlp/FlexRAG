from abc import abstractmethod
from collections import Counter

from flexrag.utils import TIME_METER

from .metrics_base import MetricsBase, METRICS


class MatchingMetrics(MetricsBase):
    name: str

    @abstractmethod
    def compute_item(self, golds: list[str], response: str) -> float:
        return

    @TIME_METER("metrics.matching_score")
    def compute(
        self, responses: list[str], golden_responses: list[list[str]], **kwargs
    ) -> tuple[float, dict[str, list[float]]]:
        matching_list = []
        for golds, response in zip(golden_responses, responses):
            matching_list.append(self.compute_item(golds, response))
        matching_score = sum(matching_list) / len(matching_list)
        return {self.name: matching_score}, {"item_score": matching_list}


@METRICS("generation_em")
class ExactMatch(MatchingMetrics):
    """ExactMatch metric computes if any of the golden responses is exactly the same as the predicted response."""

    name = "generation_em"

    def compute_item(self, golds: list[str], response: str) -> float:
        return float(response in golds)


@METRICS("generation_accuracy")
class Accuracy(MatchingMetrics):
    """Accuracy metric computes if any of the golden responses is in the predicted response."""

    name = "generation_accuracy"

    def compute_item(self, golds: list[str], response: str) -> float:
        return float(any(gold in response for gold in golds))


def f1_recall_precision(golds: list[str], response: str) -> tuple[float, float, float]:
    true_counters = [Counter(gold.split()) for gold in golds]
    pred_counter = Counter(response.split())
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


@METRICS("generation_f1")
class F1(MatchingMetrics):
    """F1 metric computes the F1 score of the predicted response against the golden responses."""

    name = "generation_f1"

    def compute_item(self, golds: list[str], response: str) -> float:
        return f1_recall_precision(golds, response)[0]


@METRICS("generation_recall")
class Recall(MatchingMetrics):
    """Recall metric computes the recall of the predicted response against the golden responses."""

    name = "generation_recall"

    def compute_item(self, golds: list[str], response: str) -> float:
        return f1_recall_precision(golds, response)[1]


@METRICS("generation_precision")
class Precision(MatchingMetrics):
    """Precision metric computes the precision of the predicted response against the golden responses."""

    name = "generation_precision"

    def compute_item(self, golds: list[str], response: str) -> float:
        return f1_recall_precision(golds, response)[2]
