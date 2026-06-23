import numpy as np

from flexrag.common import RetrievedContext, configure
from flexrag.models.scorers import PairScorerProtocol

from .ranker_base import RANKERS, RankerBase, RankerBaseConfig, RankingResult


@configure
class HFRankerConfig(RankerBaseConfig):
    """Configuration for pair-scorer-backed HuggingFace rankers.

    The scorer is an external dependency supplied to ``HFRanker`` at
    construction time. This config only controls ranking behavior inherited
    from ``RankerBaseConfig``.
    """


@RANKERS("hf", config_class=HFRankerConfig)
class HFRanker(RankerBase):
    """Rank candidates with an externally provided HuggingFace pair scorer."""

    def __init__(self, cfg: HFRankerConfig, scorer: PairScorerProtocol) -> None:
        super().__init__(cfg)
        self.scorer = scorer
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
