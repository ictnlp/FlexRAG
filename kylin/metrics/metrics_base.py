from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

from unidecode import unidecode

from kylin.text_process import normalize_answer


class MetricsBase(ABC):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--metric_normalize",
            action="store_true",
            default=False,
            help="Whether to normalize the tokens for the metric computation",
        )
        parser.add_argument(
            "--metric_lowercase",
            action="store_true",
            default=False,
            help="Whether to lowercase the text for the metric computation",
        )
        parser.add_argument(
            "--metric_unify",
            action="store_true",
            default=False,
            help="Whether to convert all character into ascii for the metric computation",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.normalize_answer = args.metric_normalize
        self.lowercase = args.metric_lowercase
        self.unify = args.metric_unify
        return

    def __call__(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> dict[str, float]:
        assert len(y_trues) == len(
            y_preds
        ), "The length of y_true and y_pred should be the same"
        y_preds = [self.preprocess_text(y) for y in y_preds]
        y_trues = [[self.preprocess_text(y_) for y_ in y] for y in y_trues]
        return self.compute(y_trues, y_preds)

    @abstractmethod
    def compute(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> tuple[float, object]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            y_trues (list[list[str]]): The true labels for each sample.
            y_preds (list[str]): The predicted labels for each sample.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        return

    def preprocess_text(self, text: str) -> str:
        if self.normalize_answer:
            text = normalize_answer(text)
        if self.unify:
            text = unidecode(text)
        if self.lowercase:
            text = text.lower()
        return text
