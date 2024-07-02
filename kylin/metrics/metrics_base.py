from abc import ABC, abstractmethod
from dataclasses import dataclass

from unidecode import unidecode

from kylin.text_process import normalize_answer


@dataclass
class MetricsConfig:
    normalize: bool = False
    lowercase: bool = False
    unify: bool = False


class MetricsBase(ABC):
    def __init__(self, cfg: MetricsConfig) -> None:
        self.args = cfg
        self.normalize_answer = cfg.normalize
        self.lowercase = cfg.lowercase
        self.unify = cfg.unify
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
