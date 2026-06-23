from typing import Any

from flexrag.common import ProgressDisplay, SimpleProgressLogger
from flexrag.common.dataclasses import RetrievedContext
from flexrag.processors.rankers.ranker_base import RankingResult


class RankerInvocation:
    """Invocation semantics for managed ranker resources."""

    def __init__(self, runtime: Any) -> None:
        """Create a ranker invocation."""
        self.runtime = runtime
        return

    async def _async_rank_core(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int,
        display: ProgressDisplay,
    ) -> RankingResult:
        with SimpleProgressLogger(
            total=1,
            interval=log_interval,
            display=display,
        ) as progress:
            result = await self.runtime.acall("async_rank", query, candidates)
            progress.update(1, desc="Ranking")
        return result

    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates synchronously."""
        run_sync = getattr(self.runtime, "run_sync", None)
        if callable(run_sync):
            return run_sync(
                self._async_rank_core(
                    query,
                    candidates,
                    log_interval,
                    display,
                )
            )

        with SimpleProgressLogger(
            total=1,
            interval=log_interval,
            display=display,
        ) as progress:
            result = self.runtime.call("rank", query, candidates)
            progress.update(1, desc="Ranking")
        return result

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        """Rank candidates asynchronously."""
        coro = self._async_rank_core(
            query,
            candidates,
            log_interval,
            display,
        )
        run_async = getattr(self.runtime, "run_async", None)
        if callable(run_async):
            return await run_async(coro)
        return await coro
