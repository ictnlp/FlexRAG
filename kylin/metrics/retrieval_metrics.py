from abc import abstractmethod
from dataclasses import dataclass

from kylin.utils import Choices

from .metrics_base import MetricsBase, MetricsConfig

try:
    from .lib_rel import get_contain_map

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


def get_contain_map_py(evidences: list[str], retrieved: list[str]) -> list[list[bool]]:
    contain_map: list[list[bool]] = []
    for ret in retrieved:
        contain_map.append([])
        for evd in evidences:
            contain_map[-1].append(evd in ret)
    return contain_map


def get_relevance_map(
    evidences: list[str],
    retrieved: list[str],
    method: str = "contains",
) -> list[list[bool]]:
    match method:
        case "contains":
            if has_librel:
                contain_map = get_contain_map(evidences, retrieved)
            else:
                contain_map = get_contain_map_py(evidences, retrieved)
        case _:
            raise ValueError(f"Invalid method: {method}")
    return contain_map


@dataclass
class SuccessRateConfig(RetrievalMetricsConfig):
    relevance_check: Choices(["contains"]) = "contains"  # type: ignore


class SuccessRate(RetrievalMetric):
    def __init__(self, cfg: SuccessRateConfig) -> None:
        super().__init__(cfg)
        self.relevance_check = cfg.relevance_check
        return

    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, dict]:
        # compute relevance map
        success_map: list[bool] = []
        for evds, rets in zip(evidences, retrieved):
            rel_map = get_relevance_map(evds, rets, self.relevance_check)
            success_map.append(any([any(i) for i in rel_map]))
        score = sum(success_map) / len(success_map)
        return score, {"success_map": success_map}


@dataclass
class RetrievalPrecisionConfig(RetrievalMetricsConfig):
    relevance_check: Choices(["contains"]) = "contains"  # type: ignore


class RetrievalPrecision(RetrievalMetric):
    def __init__(self, cfg: RetrievalPrecisionConfig) -> None:
        super().__init__(cfg)
        self.relevance_check = cfg.relevance_check
        return

    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, dict]:
        # compute relevance map
        precision_map: list[float] = []
        for evds, rets in zip(evidences, retrieved):
            rel_map = get_relevance_map(evds, rets, self.relevance_check)
            precision = sum([any(i) for i in rel_map]) / len(rets)
            precision_map.append(precision)
        score = sum(precision_map) / len(precision_map)
        return score, {"precision_map": precision_map}
