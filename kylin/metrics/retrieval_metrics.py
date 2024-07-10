from abc import abstractmethod
from dataclasses import dataclass

from kylin.utils import Choices

from .metrics_base import MetricsBase, MetricsConfig

try:
    from .lib_rel import contains_any

    has_librel = True
except:
    has_librel = False


RetrievalMetricsConfig = MetricsConfig


class RetrievalMetric(MetricsBase):
    def __call__(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> dict[str, float]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[str]]): The evidence documents.
            retrieved (list[list[str]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        assert len(evidences) == len(
            retrieved
        ), "The length of y_true and y_pred should be the same"
        evidences = [[self.preprocess_text(y_) for y_ in y] for y in evidences]
        retrieved = [[self.preprocess_text(y_) for y_ in y] for y in retrieved]
        return self.compute(evidences, retrieved)

    @abstractmethod
    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, object]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[str]]): The evidence documents.
            retrieved (list[list[str]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        return


@dataclass
class SuccessRateConfig(RetrievalMetricsConfig):
    relevance_check: Choices(["contain_any"]) = "contain_any"  # type: ignore


class SuccessRate(RetrievalMetric):
    def __init__(self, cfg: SuccessRateConfig) -> None:
        super().__init__(cfg)
        self.relevance_check = cfg.relevance_check
        return

    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, object]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[str]]): The evidence documents.
            retrieved (list[list[str]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        # compute relevance map
        relevance_map: list[bool] = []
        for evds, rets in zip(evidences, retrieved):
            relevance_map.append(self.is_relevance(evds, rets))
        score = sum(relevance_map) / len(relevance_map)
        return score, relevance_map

    def _contains_any(self, evidences: list[str], retrieved: list[str]) -> bool:
        if has_librel:
            return contains_any(evidences, retrieved)
        for evd in evidences:
            for ret in retrieved:
                if evd in ret:
                    return True
        return False

    def is_relevance(self, evidences: list[str], retrieved: list[str]):
        match self.relevance_check:
            case "contain_any":
                return self._contains_any(evidences, retrieved)
            case _:
                raise ValueError(f"Invalid relevance check: {self.relevance_check}")
