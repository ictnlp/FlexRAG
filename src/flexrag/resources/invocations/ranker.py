import inspect
from typing import Any

from flexrag.common import ProgressDisplay, SimpleProgressLogger
from flexrag.processors.rankers.ranker_base import (
    RankerCandidates,
    RankingResult,
    _build_ranking_result,
    _extract_ranking_texts,
)


class DirectRankerInvocation:
    """Invocation semantics for direct managed rankers."""

    def __init__(self, runtime: Any) -> None:
        """Create a direct ranker invocation."""
        self.runtime = runtime
        return

    def rank(
        self,
        query: str,
        candidates: RankerCandidates,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates synchronously."""
        return self.runtime.call("rank", query, candidates)

    async def async_rank(
        self,
        query: str,
        candidates: RankerCandidates,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates asynchronously."""
        result = await self.runtime.acall("async_rank", query, candidates)
        if inspect.isawaitable(result):
            return await result
        return result


class RemoteRankerInvocation:
    """Invocation semantics for remote managed rankers."""

    def __init__(
        self,
        runtime: Any,
        *,
        rank_method: str,
    ) -> None:
        """Create a remote ranker invocation.

        :param runtime: Runtime adapter used to execute primitive calls.
        :param rank_method: Primitive method for one query and candidate batch.
        """
        self.runtime = runtime
        self._rank_method = rank_method
        return

    async def _async_rank_core(
        self,
        query: str,
        candidates: RankerCandidates,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        ranking_field = await self.runtime.agetattr("ranking_field")
        reserve_num = await self.runtime.agetattr("reserve_num")
        if not candidates:
            return RankingResult(query=query, candidates=[], scores=[])

        texts = _extract_ranking_texts(candidates, ranking_field)
        with SimpleProgressLogger(total=1, interval=log_interval, display=display) as p:
            indices, scores = await self.runtime.acall(self._rank_method, query, texts)
            p.update(1, desc="Ranking")

        return _build_ranking_result(
            query=query,
            candidates=candidates,
            reserve_num=reserve_num,
            indices=indices,
            scores=scores,
        )

    async def async_rank(
        self,
        query: str,
        candidates: RankerCandidates,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates asynchronously."""
        return await self.runtime.run_async(
            self._async_rank_core(
                query,
                candidates,
                log_interval=log_interval,
                display=display,
            )
        )

    def rank(
        self,
        query: str,
        candidates: RankerCandidates,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates synchronously."""
        return self.runtime.run_sync(
            self._async_rank_core(
                query,
                candidates,
                log_interval=log_interval,
                display=display,
            )
        )
