from dataclasses import field
from math import log2
from typing import Annotated, Optional

from flexrag.common import Choices, configure, trace
from flexrag.common.dataclasses import Context, RetrievedContext
from flexrag.processors.text_processors import AnswerSimplifier

from .metrics_base import METRICS

RetrievalMeasure = Annotated[str, Choices("recall", "precision", "ndcg", "map", "mrr")]


def get_contain_map(evidences: list[str], retrieved: list[str]) -> list[list[bool]]:
    contain_map: list[list[bool]] = []
    for ret in retrieved:
        contain_map.append([])
        for evd in evidences:
            contain_map[-1].append(evd in ret)
    return contain_map


@configure
class SuccessRateConfig:
    """Configuration for ``SuccessRate`` metric.
    This metric computes whether the retrieved contexts contain any of the golden responses.

    :param eval_field: The field to evaluate. Defaults to None.
        If None, only strings are supported as the `retrieved_contexts`.
    :type eval_field: Optional[str]
    :param simplify: Whether to simplify the retrieved contexts. Defaults to True.
    :type simplify: bool
    """

    eval_field: Optional[str] = None
    simplify: bool = True


@METRICS("retrieval_success_rate", config_class=SuccessRateConfig)
class SuccessRate:
    """The SuccessRate metric computes whether the retrieved contexts contain any of the golden responses."""

    def __init__(self, cfg: SuccessRateConfig) -> None:
        self.eval_field = cfg.eval_field
        if cfg.simplify:
            self.simplifier = AnswerSimplifier()
        else:
            self.simplifier = None
        return

    @trace("metrics.retrieval_success_rate")
    def __call__(
        self,
        golden_responses: list[list[str]] = None,
        retrieved_contexts: list[list[str | Context]] = None,
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
            if self.simplifier is not None:
                ctxs = [self.simplifier(ctx) for ctx in ctxs]
                golds = [self.simplifier(gold) for gold in golds]
            rel_map = get_contain_map(golds, ctxs)
            is_success = any(sum(rel_map, []))
            success_map.append(is_success)
        score = sum(success_map) / len(success_map)
        return {"retrieval_success_rate": score}, {"success_map": success_map}


class _RetrievalMetrics:
    def __init__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
        k_values: list[int],
        measure: RetrievalMeasure,
    ) -> None:
        self.retrieved_contexts = retrieved_contexts
        self.qrels = qrels
        self.k_values = k_values
        self.measure = measure
        self.measure_specs = self._get_measure_specs()
        return

    def evaluate(self) -> tuple[dict[str, float], dict]:
        scores = {measure_str: [] for measure_str, _ in self.measure_specs}
        details: dict[str, dict[str, float]] = {}

        for n, (query_qrels, rctxs) in enumerate(
            zip(self.qrels, self.retrieved_contexts)
        ):
            query_scores = self._evaluate_query(query_qrels, rctxs)
            for measure_str, score in query_scores.items():
                scores[measure_str].append(score)
            details[str(n)] = query_scores

        return {
            measure_str: sum(values) / len(values) if len(values) > 0 else 0.0
            for measure_str, values in scores.items()
        }, details

    def _evaluate_query(
        self,
        qrels: dict[str, float],
        retrieved_contexts: list[RetrievedContext | str],
    ) -> dict[str, float]:
        retrieved: dict[str, tuple[float, int]] = {}
        for order, ctx in enumerate(retrieved_contexts):
            ctx_id = self._get_context_id(ctx)
            retrieved[ctx_id] = (self._get_retrieval_score(ctx), order)

        ranking = [
            ctx_id
            for ctx_id, _ in sorted(
                retrieved.items(), key=lambda item: (-item[1][0], item[1][1])
            )
        ]
        relevant_docs = {ctx_id for ctx_id, rel in qrels.items() if rel > 0}
        query_scores: dict[str, float] = {}
        for measure_str, k in self.measure_specs:
            match self.measure:
                case "recall":
                    score = 0.0
                    if len(relevant_docs) > 0:
                        hits = sum(ctx_id in relevant_docs for ctx_id in ranking[:k])
                        score = hits / len(relevant_docs)
                case "precision":
                    hits = sum(ctx_id in relevant_docs for ctx_id in ranking[:k])
                    score = hits / k
                case "ndcg":
                    score = self._ndcg(ranking, qrels, k)
                case "map":
                    score = self._average_precision(ranking, relevant_docs, k)
                case "mrr":
                    score = 0.0
                    for rank, ctx_id in enumerate(ranking, start=1):
                        if ctx_id in relevant_docs:
                            score = 1 / rank
                            break
                case _:
                    raise ValueError(f"Invalid measure: {self.measure}")
            query_scores[measure_str] = score
        return query_scores

    def _get_context_id(self, ctx: RetrievedContext | str) -> str:
        if isinstance(ctx, str):
            return ctx
        if ctx.context_id is None:
            raise ValueError("Retrieved contexts must have a context_id.")
        return str(ctx.context_id)

    def _get_measure_specs(self) -> list[tuple[str, int | None]]:
        if any(k <= 0 for k in self.k_values):
            raise ValueError("k_values must contain positive integers.")
        match self.measure:
            case "recall":
                if len(self.k_values) == 0:
                    raise ValueError("k_values must not be empty for recall.")
                return [(f"Recall@{k}", k) for k in self.k_values]
            case "precision":
                if len(self.k_values) == 0:
                    raise ValueError("k_values must not be empty for precision.")
                return [(f"Precision@{k}", k) for k in self.k_values]
            case "ndcg":
                if len(self.k_values) == 0:
                    return [("nDCG", None)]
                return [(f"nDCG@{k}", k) for k in self.k_values]
            case "map":
                if len(self.k_values) == 0:
                    return [("MAP", None)]
                return [(f"MAP@{k}", k) for k in self.k_values]
            case "mrr":
                return [("MRR", None)]
            case _:
                raise ValueError(f"Invalid measure: {self.measure}")

    @staticmethod
    def _get_retrieval_score(ctx: RetrievedContext | str) -> float:
        if isinstance(ctx, str):
            return 1.0
        if ctx.score is None:
            return 1.0
        return float(ctx.score)

    @staticmethod
    def _average_precision(
        ranking: list[str], relevant_docs: set[str], k: int | None = None
    ) -> float:
        if len(relevant_docs) == 0:
            return 0.0
        hits = 0
        score = 0.0
        for rank, ctx_id in enumerate(ranking[:k], start=1):
            if ctx_id in relevant_docs:
                hits += 1
                score += hits / rank
        return score / len(relevant_docs)

    @staticmethod
    def _dcg(relevances: list[float]) -> float:
        return sum(
            (2**rel - 1) / log2(rank + 1) for rank, rel in enumerate(relevances, 1)
        )

    @staticmethod
    def _ndcg(
        ranking: list[str], qrels: dict[str, float], k: int | None = None
    ) -> float:
        ideal_rels = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)
        if k is not None:
            ideal_rels = ideal_rels[:k]
        if len(ideal_rels) == 0:
            return 0.0
        idcg = _RetrievalMetrics._dcg(ideal_rels)
        if idcg == 0:
            return 0.0
        ranking_rels = [qrels.get(ctx_id, 0.0) for ctx_id in ranking[:k]]
        return _RetrievalMetrics._dcg(ranking_rels) / idcg


@configure
class RetrievalRecallConfig:
    """Configuration for ``RetrievalRecall`` metric.
    This metric computes the recall of the retrieved contexts.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_recall", config_class=RetrievalRecallConfig)
class RetrievalRecall:
    """The RetrievalRecall metric computes the recall of the retrieved contexts."""

    def __init__(self, cfg: RetrievalRecallConfig) -> None:
        self.k_values = cfg.k_values
        return

    @trace("metrics.retrieval_recall")
    def __call__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
    ) -> tuple[dict[str, float], dict]:
        return _RetrievalMetrics(
            retrieved_contexts=retrieved_contexts,
            qrels=qrels,
            k_values=self.k_values,
            measure="recall",
        ).evaluate()


@configure
class RetrievalPrecisionConfig:
    """Configuration for ``RetrievalPrecision`` metric.
    This metric computes the precision of the retrieved contexts.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_precision", config_class=RetrievalPrecisionConfig)
class RetrievalPrecision:
    """The RetrievalPrecision metric computes the precision of the retrieved contexts."""

    def __init__(self, cfg: RetrievalPrecisionConfig) -> None:
        self.k_values = cfg.k_values
        return

    @trace("metrics.retrieval_precision")
    def __call__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
    ) -> tuple[dict[str, float], dict]:
        return _RetrievalMetrics(
            retrieved_contexts=retrieved_contexts,
            qrels=qrels,
            k_values=self.k_values,
            measure="precision",
        ).evaluate()


@configure
class RetrievalMAPConfig:
    """Configuration for ``RetrievalMAP`` metric.
    This metric computes the MAP of the retrieved contexts.

    :param k_values: The k values for evaluation. Defaults to [].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=list)


@METRICS("retrieval_map", config_class=RetrievalMAPConfig)
class RetrievalMAP:
    """The RetrievalMAP metric computes the Mean Average Precision (MAP) of the retrieved contexts."""

    def __init__(self, cfg: RetrievalMAPConfig) -> None:
        self.k_values = cfg.k_values
        return

    @trace("metrics.retrieval_map")
    def __call__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
    ) -> tuple[dict[str, float], dict]:
        return _RetrievalMetrics(
            retrieved_contexts=retrieved_contexts,
            qrels=qrels,
            k_values=self.k_values,
            measure="map",
        ).evaluate()


@configure
class RetrievalNDCGConfig:
    """Configuration for ``RetrievalNDCG`` metric.
    This metric computes the nDCG of the retrieved contexts.

    :param k_values: The k values for evaluation. Defaults to [].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=list)


@METRICS("retrieval_ndcg", config_class=RetrievalNDCGConfig)
class RetrievalNDCG:
    """The RetrievalNDCG metric computes the Normalized Discounted Cumulative Gain (nDCG) of the retrieved contexts."""

    def __init__(self, cfg: RetrievalNDCGConfig) -> None:
        self.k_values = cfg.k_values
        return

    @trace("metrics.retrieval_ndcg")
    def __call__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
    ) -> tuple[dict[str, float], dict]:
        return _RetrievalMetrics(
            retrieved_contexts=retrieved_contexts,
            qrels=qrels,
            k_values=self.k_values,
            measure="ndcg",
        ).evaluate()


@METRICS("retrieval_mrr")
class RetrievalMRR:
    """The RetrievalMRR metric computes the Mean Reciprocal Rank (MRR) of the retrieved contexts."""

    @trace("metrics.retrieval_mrr")
    def __call__(
        self,
        retrieved_contexts: list[list[RetrievedContext | str]],
        qrels: list[dict[str, float]],
    ) -> tuple[dict[str, float], dict]:
        return _RetrievalMetrics(
            retrieved_contexts=retrieved_contexts,
            qrels=qrels,
            k_values=[],
            measure="mrr",
        ).evaluate()
