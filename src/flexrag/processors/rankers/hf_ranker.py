import numpy as np

from flexrag.common import RetrievedContext, configure
from flexrag.models.scorers import (
    HFColBertScorer,
    HFColBertScorerConfig,
    HFCrossEncoderScorer,
    HFCrossEncoderScorerConfig,
    HFLogitsScorer,
    HFLogitsScorerConfig,
)

from .ranker_base import RANKERS, RankerBase, RankerBaseConfig, RankingResult


class _HFScorerRankerBase(RankerBase):
    scorer_cls = None

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        if self.scorer_cls is None:
            raise ValueError(
                f"{self.__class__.__name__}.scorer_cls must be configured."
            )
        self.scorer = self.scorer_cls(cfg)
        return

    def _prepare_pairs(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> list[tuple[str, str]]:
        if candidates and isinstance(candidates[0], RetrievedContext):
            assert self.ranking_field is not None, (
                "ranking_field must be specified when ranking RetrievedContext"
            )
        return [
            (query, cand if isinstance(cand, str) else cand.data[self.ranking_field])
            for cand in candidates
        ]

    def _build_result(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        scores: np.ndarray,
    ) -> RankingResult:
        ranked_indices = scores.argsort()[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices].tolist()
        if self.reserve_num > 0:
            ranked_candidates = ranked_candidates[: self.reserve_num]
            ranked_scores = ranked_scores[: self.reserve_num]
        return RankingResult(
            query=query,
            candidates=ranked_candidates,
            scores=ranked_scores,
        )

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        pairs = self._prepare_pairs(query, candidates)
        scores = self.scorer.score(pairs)
        return self._build_result(query, candidates, scores)

    async def async_rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        pairs = self._prepare_pairs(query, candidates)
        scores = await self.scorer.async_score(pairs)
        return self._build_result(query, candidates, scores)

    def close(self) -> None:
        close = getattr(self.scorer, "close", None)
        if callable(close):
            close()
        return


@configure
class HFColBertRankerConfig(RankerBaseConfig, HFColBertScorerConfig):
    """The configuration for the HuggingFace ColBERT ranker."""


@RANKERS("hf_colbert", config_class=HFColBertRankerConfig)
class HFColBertRanker(_HFScorerRankerBase):
    """HFColBertRanker: The ranker based on the HuggingFace ColBERT model."""

    scorer_cls = HFColBertScorer


@configure
class HFCrossEncoderRankerConfig(RankerBaseConfig, HFCrossEncoderScorerConfig):
    """The configuration for the HuggingFace CrossEncoder ranker."""


@RANKERS("hf_crossencoder", config_class=HFCrossEncoderRankerConfig)
class HFCrossEncoderRanker(_HFScorerRankerBase):
    """HFCrossEncoderRanker: The ranker based on the HuggingFace CrossEncoder model."""

    scorer_cls = HFCrossEncoderScorer


@configure
class HFLogitsRankerConfig(RankerBaseConfig, HFLogitsScorerConfig):
    """The configuration for the HuggingFace Logits ranker."""


@RANKERS("hf_logits", config_class=HFLogitsRankerConfig)
class HFLogitsRanker(_HFScorerRankerBase):
    """HFLogitsRanker: The ranker based on the HuggingFace Logits model."""

    scorer_cls = HFLogitsScorer
