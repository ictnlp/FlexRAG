from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kylin.text_process import PipelineConfig, Pipeline
from kylin.utils import TimeMeter


@dataclass
class MetricsConfig:
    answer_preprocess_pipeline: PipelineConfig = field(default_factory=PipelineConfig)  # type: ignore


class MetricsBase(ABC):
    def __init__(self, cfg: MetricsConfig) -> None:
        self.args = cfg
        self.preprocess_pipeline = Pipeline(cfg.answer_preprocess_pipeline)
        return

    @TimeMeter("metric")
    def __call__(
        self, y_trues: list[list[str]], y_preds: list[str]
    ) -> dict[str, float]:
        assert len(y_trues) == len(
            y_preds
        ), "The length of y_true and y_pred should be the same"
        y_preds = [self.preprocess_pipeline(y) for y in y_preds]
        y_trues = [[self.preprocess_pipeline(y_) for y_ in y] for y in y_trues]
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
