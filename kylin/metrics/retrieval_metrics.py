from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from sacrebleu import sentence_bleu

from kylin.retriever import RetrievedContext
from kylin.utils import Choices

from .metrics_base import MetricsBase, MetricsConfig

try:
    from .lib_rel import get_contain_map

    has_librel = True
except:
    has_librel = False


class RetrievalMetricsConfig(MetricsConfig):
    evaluate_field: Optional[str] = None


class RetrievalMetric(MetricsBase):
    cfg: RetrievalMetricsConfig

    def __call__(
        self,
        evidences: list[list[Any | RetrievedContext]],
        retrieved: list[list[Any | RetrievedContext]],
    ) -> dict[str, float]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[RetrievedContext]]): The evidence documents.
            retrieved (list[list[RetrievedContext]]): The retrieved documents.

        Returns:
            tuple[float, object]: A tuple containing the metric value and additional metric-specific information.
        """
        assert len(evidences) == len(
            retrieved
        ), "The length of evidences and retrieved contexts should be the same"
        if isinstance(evidences[0][0], RetrievedContext):
            assert (
                self.cfg.evaluate_field is not None
            ), "The evaluate_field should be provided when the input is RetrievedContext."
            evidences_ = [
                [i.data[self.cfg.evaluate_field] for i in j] for j in evidences
            ]
        else:
            evidences_ = evidences
        if isinstance(retrieved[0][0], RetrievedContext):
            assert (
                self.cfg.evaluate_field is not None
            ), "The evaluate_field should be provided when the input is RetrievedContext."
            retrieved_ = [
                [i.data[self.cfg.evaluate_field] for i in j] for j in retrieved
            ]
        else:
            retrieved_ = retrieved
        return self.compute(evidences_, retrieved_)

    @abstractmethod
    def compute(
        self, evidences: list[list[Any]], retrieved: list[list[Any]]
    ) -> tuple[float, object]:
        """
        Compute the metric value and additional metric-specific information.

        Args:
            evidences (list[list[Any]]): The evidence documents.
            retrieved (list[list[Any]]): The retrieved documents.

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


def get_bleu_map(evidences: list[str], retrieved: list[str]) -> list[list[float]]:
    bleu_map: list[list[float]] = []
    for ret in retrieved:
        bleu_map.append([])
        for evd in evidences:
            bleu_map[-1].append(sentence_bleu(ret, evd).score)
    return bleu_map


def get_relevance_map(
    evidences: list[str],
    retrieved: list[str],
    method: str = "contains",
) -> list[list[float]]:
    match method:
        case "contains":
            if has_librel:
                contain_map = get_contain_map(evidences, retrieved)
            else:
                contain_map = get_contain_map_py(evidences, retrieved)
        case "bleu":
            contain_map = get_bleu_map(evidences, retrieved)
        case _:
            raise ValueError(f"Invalid method: {method}")
    return contain_map


@dataclass
class SuccessRateConfig(RetrievalMetricsConfig):
    relevance_check: Choices(["contains", "bleu"]) = "contains"  # type: ignore
    minimum_relevance: float = 0.2


class SuccessRate(RetrievalMetric):
    def __init__(self, cfg: SuccessRateConfig) -> None:
        super().__init__(cfg)
        self.relevance_check = cfg.relevance_check
        self.min_rel = cfg.minimum_relevance
        return

    def compute(
        self, evidences: list[list[str]], retrieved: list[list[str]]
    ) -> tuple[float, dict]:
        # compute relevance map
        success_map: list[bool] = []
        for evds, rets in zip(evidences, retrieved):
            rel_map = get_relevance_map(evds, rets, self.relevance_check)
            is_success = any([any([i > self.min_rel for i in j]) for j in rel_map])
            success_map.append(is_success)
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
            precision = sum([any(i) for i in rel_map]) / max(len(rets), 1)
            precision_map.append(precision)
        score = sum(precision_map) / len(precision_map)
        return score, {"precision_map": precision_map}
