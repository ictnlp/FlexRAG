from abc import ABC, abstractmethod
from dataclasses import dataclass

from kylin.utils import TIME_METER


@dataclass
class MetricsConfig: ...


class MetricsBase(ABC):
    def __init__(self, cfg: MetricsConfig) -> None:
        self.cfg = cfg
        return

    @TIME_METER("metric")
    def __call__(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> dict[str, float]:
        assert len(y_trues) == len(
            y_preds
        ), "The length of y_true and y_pred should be the same"
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
