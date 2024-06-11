from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace


class MetricsBase(ABC):
    @abstractmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        return parser

    def __init__(self, args: Namespace) -> None:
        self.args = args
        return

    def __call__(
        self, y_trues: list[str], y_preds: list[list[str]]
    ) -> dict[str, float]:
        assert len(y_trues) == len(
            y_preds
        ), "The length of y_true and y_pred should be the same"
        y_trues, y_preds = self.preprocess(y_trues, y_preds)
        return self.compute(y_trues, y_preds)

    @abstractmethod
    def compute(self, y_trues: list[list[str]], y_preds: list[str]):
        return

    def preprocess(self, y_trues: list[list[str]], y_preds: list[str]):
        return y_trues, y_preds
