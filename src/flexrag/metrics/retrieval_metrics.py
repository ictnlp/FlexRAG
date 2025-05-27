from collections import defaultdict
from dataclasses import field
from typing import Annotated, Optional

import pytrec_eval

from flexrag.text_process import AnswerSimplifier
from flexrag.utils import TIME_METER, Choices, configure
from flexrag.utils.dataclasses import Context, RetrievedContext

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
class SuccessRate(MetricsBase):
    """The SuccessRate metric computes whether the retrieved contexts contain any of the golden responses."""

    def __init__(self, cfg: SuccessRateConfig) -> None:
        self.eval_field = cfg.eval_field
        if cfg.simplify:
            self.simplifier = AnswerSimplifier()
        else:
            self.simplifier = None
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
            if self.simplifier is not None:
                ctxs = [self.simplifier(ctx) for ctx in ctxs]
                golds = [self.simplifier(gold) for gold in golds]
            rel_map = get_contain_map_py(golds, ctxs)
            is_success = any(sum(rel_map, []))
            success_map.append(is_success)
        score = sum(success_map) / len(success_map)
        return {"retrieval_success_rate": score}, {"success_map": success_map}


def pytrec_evaluate(
    retrieved_contexts: list[list[RetrievedContext | str]],
    golden_contexts: list[list[Context | str]],
    k_values: list[int] = [1, 5, 10],
    measure: Annotated[
        str, Choices("recall", "precision", "ndcg", "map", "mrr")
    ] = "recall",
) -> tuple[dict[str, float], dict]:
    """Evaluate the retrieval results using pytrec_eval.

    :param retrieved_contexts: The retrieved contexts.
    :type retrieved_contexts: list[list[RetrievedContext]]
    :param golden_contexts: The golden contexts.
    :type golden_contexts: list[list[Context]]
    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int], optional
    :param measure: The evaluation measure. Defaults to "recall".
        Available choices are "recall", "precision", "ndcg", and "map".
    :type measure: str, optional
    :return: The evaluation scores and details.
    :rtype: tuple[dict[str, float], dict]
    """
    # convert flexrag format to pytrec_eval format
    qrels: dict[str, dict[str, int]] = {}
    retrieved: dict[str, dict[str, float]] = {}
    ctx_ids: dict[str, str] = {}  # label the context if it is a string
    for n, (gctxs, rctxs) in enumerate(zip(golden_contexts, retrieved_contexts)):
        qrels[str(n)] = {}
        retrieved[str(n)] = {}
        for ctx in gctxs:
            if isinstance(ctx, str):
                ctx_id = ctx_ids.get(ctx, str(len(ctx_ids)))
                ctx_ids[ctx] = ctx_id
                ctx_score = 1
            else:
                ctx_id = ctx.context_id
                ctx_score = ctx.meta_data.get("score", 1)
            qrels[str(n)][ctx_id] = ctx_score
        for ctx in rctxs:
            if isinstance(ctx, str):
                ctx_id = ctx_ids.get(ctx, str(len(ctx_ids)))
                ctx_ids[ctx] = ctx_id
                ctx_score = 1.0
            else:
                ctx_id = ctx.context_id
                ctx_score = ctx.score or 1.0
            retrieved[str(n)][ctx_id] = ctx_score

    # prepare pytrec_eval measure_strings
    k_values = [str(k) for k in k_values]
    measures_: set[str]  # for pytrec_eval args
    measure_strs_: list[str]  # for pytrec_eval results
    measure_strs: list[str]  # for FlexRAG results
    match measure:
        case "recall":
            measures_ = {f"recall." + ",".join(k_values)}
            measure_strs_ = [f"recall_{k}" for k in k_values]
            measure_strs = [f"Recall@{k}" for k in k_values]
        case "precision":
            measures_ = {f"P." + ",".join(k_values)}
            measure_strs_ = [f"P_{k}" for k in k_values]
            measure_strs = [f"Precision@{k}" for k in k_values]
        case "ndcg":
            if len(k_values) > 1:
                measures_ = {f"ndcg_cut." + ",".join(k_values)}
                measure_strs_ = [f"ndcg_cut_{k}" for k in k_values]
                measure_strs = [f"nDCG@{k}" for k in k_values]
            else:
                measures_ = {"ndcg"}
                measure_strs_ = ["ndcg"]
                measure_strs = ["nDCG"]
        case "map":
            if len(k_values) > 1:
                measures_ = {f"map_cut." + ",".join(k_values)}
                measure_strs_ = [f"map_cut_{k}" for k in k_values]
                measure_strs = [f"MAP@{k}" for k in k_values]
            else:
                measures_ = {"map"}
                measure_strs_ = ["map"]
                measure_strs = ["MAP"]
        case "mrr":
            measures_ = {"recip_rank"}
            measure_strs_ = ["recip_rank"]
            measure_strs = ["MRR"]
        case _:
            raise ValueError(f"Invalid measure: {measure}")

    # evaluate using pytrec_eval
    evaluator = pytrec_eval.RelevanceEvaluator(
        query_relevance=qrels,
        measures=measures_,
    )
    details = evaluator.evaluate(retrieved)

    # extract the results
    scores = defaultdict(list)
    for key in retrieved.keys():
        for measure_str_, measure_str in zip(measure_strs_, measure_strs):
            scores[measure_str].append(details[key][measure_str_])
    scores = {k: sum(v) / len(v) for k, v in scores.items()}
    return scores, details


@configure
class RetrievalRecallConfig:
    """Configuration for ``RetrievalRecall`` metric.
    This metric computes the recall of the retrieved contexts.
    The computation is based on `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_recall", config_class=RetrievalRecallConfig)
class RetrievalRecall(MetricsBase):
    """The RetrievalRecall metric computes the recall of the retrieved contexts."""

    def __init__(self, cfg: RetrievalRecallConfig) -> None:
        self.k_values = cfg.k_values
        return

    @TIME_METER("metrics.retrieval_recall")
    def compute(
        self,
        retrieved_contexts: list[list[RetrievedContext]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        scores, details = pytrec_evaluate(
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
            k_values=self.k_values,
            measure="recall",
        )
        return scores, details


@configure
class RetrievalPrecisionConfig:
    """Configuration for ``RetrievalPrecision`` metric.
    This metric computes the precision of the retrieved contexts.
    The computation is based on `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=lambda: [1, 5, 10])


@METRICS("retrieval_precision", config_class=RetrievalPrecisionConfig)
class RetrievalPrecision(MetricsBase):
    """The RetrievalPrecision metric computes the precision of the retrieved contexts."""

    def __init__(self, cfg: RetrievalPrecisionConfig) -> None:
        self.k_values = cfg.k_values
        return

    @TIME_METER("metrics.retrieval_precision")
    def compute(
        self,
        retrieved_contexts: list[list[RetrievedContext]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[float, object]:
        scores, details = pytrec_evaluate(
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
            k_values=self.k_values,
            measure="precision",
        )
        return scores, details


@configure
class RetrievalMAPConfig:
    """Configuration for ``RetrievalMAP`` metric.
    This metric computes the MAP of the retrieved contexts.
    The computation is based on `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=list)


@METRICS("retrieval_map", config_class=RetrievalMAPConfig)
class RetrievalMAP(MetricsBase):
    """The RetrievalMAP metric computes the Mean Average Precision (MAP) of the retrieved contexts."""

    def __init__(self, cfg: RetrievalMAPConfig) -> None:
        self.k_values = cfg.k_values
        return

    @TIME_METER("metrics.retrieval_map")
    def compute(
        self,
        retrieved_contexts: list[list[RetrievedContext]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        scores, details = pytrec_evaluate(
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
            k_values=self.k_values,
            measure="map",
        )
        return scores, details


@configure
class RetrievalNDCGConfig:
    """Configuration for ``RetrievalNDCG`` metric.
    This metric computes the nDCG of the retrieved contexts.
    The computation is based on `pytrec_eval <https://github.com/cvangysel/pytrec_eval>`_.

    :param k_values: The k values for evaluation. Defaults to [1, 5, 10].
    :type k_values: list[int]
    """

    k_values: list[int] = field(default_factory=list)


@METRICS("retrieval_ndcg", config_class=RetrievalNDCGConfig)
class RetrievalNDCG(MetricsBase):
    """The RetrievalNDCG metric computes the Normalized Discounted Cumulative Gain (nDCG) of the retrieved contexts."""

    def __init__(self, cfg: RetrievalNDCGConfig) -> None:
        self.k_values = cfg.k_values
        return

    @TIME_METER("metrics.retrieval_ndcg")
    def compute(
        self,
        retrieved_contexts: list[list[RetrievedContext]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        scores, details = pytrec_evaluate(
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
            k_values=self.k_values,
            measure="ndcg",
        )
        return scores, details


@METRICS("retrieval_mrr")
class RetrievalMRR(MetricsBase):
    """The RetrievalMRR metric computes the Mean Reciprocal Rank (MRR) of the retrieved contexts."""

    @TIME_METER("metrics.retrieval_mrr")
    def compute(
        self,
        retrieved_contexts: list[list[RetrievedContext]] = None,
        golden_contexts: list[list[Context]] = None,
        **kwargs,
    ) -> tuple[dict[str, float], dict]:
        scores, details = pytrec_evaluate(
            retrieved_contexts=retrieved_contexts,
            golden_contexts=golden_contexts,
            measure="mrr",
        )
        return scores, details
