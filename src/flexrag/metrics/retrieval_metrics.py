from dataclasses import dataclass, field
from typing import Optional

from flexrag.common_dataclass import Context
from flexrag.data import TextProcessPipeline, TextProcessPipelineConfig
from flexrag.utils import TIME_METER

from .metrics_base import METRICS, MetricsBase

try:
    from .lib_rel import get_contain_map

    has_librel = True
except:
    has_librel = False


def get_contain_map_py(evidences: list[str], retrieved: list[str]) -> list[list[bool]]:
    if has_librel:
        return get_contain_map(evidences, retrieved)
    contain_map: list[list[bool]] = []
    for ret in retrieved:
        contain_map.append([])
        for evd in evidences:
            contain_map[-1].append(evd in ret)
    return contain_map


@dataclass
class SuccessRateConfig:
    """Configuration for SuccessRate metric.

    :param eval_field: The field to evaluate. Defaults to None.
        If None, only strings are supported as the `retrieved_contexts`.
    :type eval_field: Optional[str]
    :param context_preprocess: The preprocessing pipeline for the context. Defaults to TextProcessPipelineConfig.
    :type context_preprocess: TextProcessPipelineConfig
    """

    eval_field: Optional[str] = None
    context_preprocess: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore


@METRICS("retrieval_success_rate", config_class=SuccessRateConfig)
class SuccessRate(MetricsBase):
    """SuccessRate metric.

    This metric computes whether the retrieved contexts contain any of the golden responses.
    """

    def __init__(self, cfg: SuccessRateConfig) -> None:
        self.eval_field = cfg.eval_field
        self.context_pipeline = TextProcessPipeline(cfg.context_preprocess)
        return

    @TIME_METER("metrics.retrieval_success_rate")
    def compute(
        self,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        # compute relevance map
        success_map: list[bool] = []
        for golds, ctxs in zip(golden_responses, retrieved_contexts):
            if len(ctxs) == 0:
                success_map.append(False)
                continue
            if isinstance(ctxs[0], Context):
                assert self.eval_field is not None
                ctxs = [ctx.data[self.eval_field] for ctx in ctxs]
            if isinstance(ctxs[0], dict):
                ctxs = [ctx["data"][self.eval_field] for ctx in ctxs]
            ctxs = [self.context_pipeline(ctx) for ctx in ctxs]
            rel_map = get_contain_map_py(golds, ctxs)
            is_success = any(sum(rel_map, []))
            success_map.append(is_success)
        score = sum(success_map) / len(success_map)
        return {"retrieval_success_rate": score}, {"success_map": success_map}


def compute_recall_precision(
    retrieved: list[Context],
    golden: list[Context],
    top_k: int = -1,
) -> dict[str, float]:
    retrieved_ids = [ctx.context_id for ctx in retrieved[:top_k]]
    golden_ids = [ctx.context_id for ctx in golden]
    succ_num = sum(gold in retrieved_ids for gold in golden_ids)
    return {
        "recall": succ_num / len(golden_ids),
        "precision": succ_num / len(retrieved_ids),
    }


@dataclass
class RetrievalRecallConfig:
    top_k: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_recall", config_class=RetrievalRecallConfig)
class RetrievalRecall(MetricsBase):
    """RetrievalRecall metric.

    This metric computes the recall of the retrieved contexts."""

    def __init__(self, cfg: RetrievalRecallConfig) -> None:
        self.top_k = cfg.top_k
        return

    @TIME_METER("metrics.retrieval_recall")
    def compute(
        self,
        retrieved_contexts: list[list[Context]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        scores = {}
        details = {}
        for top_k in self.top_k:
            name = f"recall@{top_k}" if top_k != -1 else "recall"
            item_scores = [
                compute_recall_precision(retrieved, golden, top_k=top_k)
                for retrieved, golden in zip(retrieved_contexts, golden_contexts)
            ]
            scores[name] = sum(score["recall"] for score in item_scores) / len(
                item_scores
            )
            details[name] = item_scores
        return scores, details


@dataclass
class RetrievalPrecisionConfig:
    top_k: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_precision", config_class=RetrievalPrecisionConfig)
class RetrievalPrecision(MetricsBase):
    """RetrievalPrecision metric.

    This metric computes the precision of the retrieved contexts."""

    def __init__(self, cfg: RetrievalPrecisionConfig) -> None:
        self.top_k = cfg.top_k
        return

    @TIME_METER("metrics.retrieval_precision")
    def compute(
        self,
        retrieved_contexts: list[list[Context]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[float, object]:
        scores = {}
        details = {}
        for top_k in self.top_k:
            name = f"precision@{top_k}" if top_k != -1 else "precision"
            item_scores = [
                compute_recall_precision(retrieved, golden, top_k=1)
                for retrieved, golden in zip(retrieved_contexts, golden_contexts)
            ]
            scores[name] = sum(score["precision"] for score in item_scores) / len(
                item_scores
            )
            details[name] = item_scores
        return scores, details
