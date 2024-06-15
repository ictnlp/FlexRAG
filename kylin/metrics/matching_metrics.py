from abc import abstractmethod
from collections import Counter

from .metrics_base import MetricsBase


class MatchingMetrics(MetricsBase):
    @abstractmethod
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return

    def compute(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[float, dict[str, list[float]]]:
        matching_list = []
        for y_t, y_p in zip(y_trues, y_preds):
            matching_list.append(self.compute_item(y_t, y_p))
        matching_score = sum(matching_list) / len(matching_list)
        return matching_score, {"item_score": matching_list}


class ExactMatch(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return float(y_pred in y_trues)


class Accuracy(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return float(any(y_t in y_pred for y_t in y_trues))


def f1_recall_precision(y_trues: list[str], y_pred: str) -> tuple[float, float, float]:
    true_counters = [Counter(y_t.split()) for y_t in y_trues]
    pred_counter = Counter(y_pred.split())
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    for y_t in true_counters:
        common = sum((y_t & pred_counter).values())
        if common == 0:
            continue
        p = 1.0 * common / sum(pred_counter.values())
        r = 1.0 * common / sum(y_t.values())
        f1_ = (2 * p * r) / (p + r)
        precision = max(p, precision)
        recall = max(r, recall)
        f1 = max(f1, f1_)
    return f1, recall, precision


class F1(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return f1_recall_precision(y_trues, y_pred)[0]


class Recall(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return f1_recall_precision(y_trues, y_pred)[1]


class Precision(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return f1_recall_precision(y_trues, y_pred)[2]
