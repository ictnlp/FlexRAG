from __future__ import annotations

from flexrag.common.dataclasses import RetrievedContext
from flexrag.processors.rankers.ranker_base import RankingResult

from ..runtime import RuntimeCall
from .base import TypedHandle


class RankerHandle(TypedHandle):
    """Typed proxy for ranker resources.

    Rankers operate on one query and one candidate list per public call, so the
    handle submits a single primitive runtime call. It does not own the ranker
    lifecycle.
    """

    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: str = "auto",
    ) -> RankingResult:
        """Synchronously rank candidates for one query.

        :param query: Query text.
        :param candidates: Candidate strings or retrieved contexts.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Ranking result returned by the raw ranker.
        """
        return self._target.batch_call(
            [RuntimeCall("rank", args=(query, candidates), weight=1)],
            log_interval=log_interval,
            display=display,
            desc="Ranking",
        )[0]

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: str = "auto",
    ) -> RankingResult:
        """Asynchronously rank candidates for one query.

        :param query: Query text.
        :param candidates: Candidate strings or retrieved contexts.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Ranking result returned by the raw ranker.
        """
        return (
            await self._target.async_batch_call(
                [RuntimeCall("async_rank", args=(query, candidates), weight=1)],
                log_interval=log_interval,
                display=display,
                desc="Ranking",
            )
        )[0]
