from flexrag.common import RetrievedContext, configure
from flexrag.models.scorers import PairScorerProtocol

from .ranker_base import (
    RANKERS,
    RankerBase,
    RankerBaseConfig,
    RankingResult,
    _build_ranking_result,
    _extract_ranking_texts,
)


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

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        texts = _extract_ranking_texts(candidates, self.ranking_field)
        pairs = [(query, text) for text in texts]
        scores = self.scorer.score(pairs)
        return _build_ranking_result(
            query,
            candidates,
            reserve_num=self.reserve_num,
            indices=None,
            scores=scores,
        )

    async def async_rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        texts = _extract_ranking_texts(candidates, self.ranking_field)
        pairs = [(query, text) for text in texts]
        scores = await self.scorer.async_score(pairs)
        return _build_ranking_result(
            query,
            candidates,
            reserve_num=self.reserve_num,
            indices=None,
            scores=scores,
        )
