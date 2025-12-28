from flexrag.common import RetrievedContext
from flexrag.models.scorers import (
    HFColBertScorer,
    HFColBertScorerConfig,
    HFCrossEncoderScorer,
    HFCrossEncoderScorerConfig,
    HFLogitsScorer,
    HFLogitsScorerConfig,
)

from .ranker_base import RANKERS, RankerBase, RankerBaseConfig, RankingResult


class HFColBertRankerConfig(RankerBaseConfig, HFColBertScorerConfig):
    """The configuration for the HuggingFace ColBERT ranker."""


@RANKERS("hf_colbert", config_class=HFColBertRankerConfig)
class HFColBertRanker(RankerBase):
    """HFColBertRanker: The ranker based on the HuggingFace ColBERT model."""

    def __init__(self, cfg: HFColBertRankerConfig) -> None:
        super().__init__(cfg)
        self.scorer = HFColBertScorer(cfg)
        return

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        # score the candidates
        if isinstance(candidates[0], RetrievedContext):
            assert (
                self.ranking_field is not None
            ), "ranking_field must be specified when ranking RetrievedContext"
        pairs = [
            (query, cand if isinstance(cand, str) else cand.data[self.ranking_field])
            for cand in candidates
        ]
        scores = self.scorer.score(pairs)
        # rank the candidates
        ranked_indices = scores.argsort()[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices].tolist()
        # reserve top-k candidates
        if self.reserve_num > 0:
            ranked_candidates = ranked_candidates[: self.reserve_num]
            ranked_scores = ranked_scores[: self.reserve_num]
        return RankingResult(
            query=query,
            candidates=ranked_candidates,
            scores=ranked_scores,
        )


class HFCrossEncoderRankerConfig(RankerBaseConfig, HFCrossEncoderScorerConfig):
    """The configuration for the HuggingFace CrossEncoder ranker."""


@RANKERS("hf_crossencoder", config_class=HFCrossEncoderRankerConfig)
class HFCrossEncoderRanker(RankerBase):
    """HFCrossEncoderRanker: The ranker based on the HuggingFace CrossEncoder model."""

    def __init__(self, cfg: HFCrossEncoderRankerConfig) -> None:
        super().__init__(cfg)
        self.scorer = HFCrossEncoderScorer(cfg)
        return

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        # score the candidates
        if isinstance(candidates[0], RetrievedContext):
            assert (
                self.ranking_field is not None
            ), "ranking_field must be specified when ranking RetrievedContext"
        pairs = [
            (query, cand if isinstance(cand, str) else cand.data[self.ranking_field])
            for cand in candidates
        ]
        scores = self.scorer.score(pairs)
        # rank the candidates
        ranked_indices = scores.argsort()[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices].tolist()
        # reserve top-k candidates
        if self.reserve_num > 0:
            ranked_candidates = ranked_candidates[: self.reserve_num]
            ranked_scores = ranked_scores[: self.reserve_num]
        return RankingResult(
            query=query,
            candidates=ranked_candidates,
            scores=ranked_scores,
        )


class HFLogitsRankerConfig(RankerBaseConfig, HFLogitsScorerConfig):
    """The configuration for the HuggingFace Logits ranker."""


@RANKERS("hf_logits", config_class=HFLogitsRankerConfig)
class HFLogitsRanker(RankerBase):
    """HFLogitsRanker: The ranker based on the HuggingFace Logits model."""

    def __init__(self, cfg: HFLogitsRankerConfig) -> None:
        super().__init__(cfg)
        self.scorer = HFLogitsScorer(cfg)
        return

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        # score the candidates
        if isinstance(candidates[0], RetrievedContext):
            assert (
                self.ranking_field is not None
            ), "ranking_field must be specified when ranking RetrievedContext"
        pairs = [
            (query, cand if isinstance(cand, str) else cand.data[self.ranking_field])
            for cand in candidates
        ]
        scores = self.scorer.score(pairs)
        # rank the candidates
        ranked_indices = scores.argsort()[::-1]
        ranked_candidates = [candidates[i] for i in ranked_indices]
        ranked_scores = scores[ranked_indices].tolist()
        # reserve top-k candidates
        if self.reserve_num > 0:
            ranked_candidates = ranked_candidates[: self.reserve_num]
            ranked_scores = ranked_scores[: self.reserve_num]
        return RankingResult(
            query=query,
            candidates=ranked_candidates,
            scores=ranked_scores,
        )
