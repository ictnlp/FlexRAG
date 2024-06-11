from argparse import ArgumentParser, Namespace
from abc import abstractmethod
from collections import Counter
import numpy as np

from .metrics_base import MetricsBase
from kylin.text_process import normalize_token


class MatchingMetrics(MetricsBase):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--normalize_token",
            action="store_true",
            default=False,
            help="Whether to normalize the token",
        )
        parser.add_argument(
            "--lowercase",
            action="store_true",
            default=False,
            help="Whether to lowercase the text",
        )
        return parser

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.normalize_token = args.normalize_token
        self.lowercase = args.lowercase
        return

    @abstractmethod
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return

    def compute(self, y_trues: list[list[str]], y_preds: list[str]) -> float:
        matching_list = []
        for y_t, y_p in zip(y_trues, y_preds):
            matching_list.append(self.compute_item(y_t, y_p))
        matching_list = np.array(matching_list)
        return matching_list.mean()

    def preprocess(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[list[str], list[list[str]]]:
        if self.normalize_token:
            y_preds = [normalize_token(y) for y in y_preds]
            y_trues = [[normalize_token(y) for y in y_t] for y_t in y_trues]
        if self.lowercase:
            y_preds = [y.lower() for y in y_preds]
            y_trues = [[y.lower() for y in y_t] for y_t in y_trues]
        return y_trues, y_preds


class ExactMatch(MatchingMetrics):
    def compute_item(self, y_trues: list[str], y_pred: str) -> float:
        return float(y_pred in y_trues)


class ContainsMatch(MatchingMetrics):
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
