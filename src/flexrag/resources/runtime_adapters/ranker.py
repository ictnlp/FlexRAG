import inspect
from typing import Any

from flexrag.processors.rankers.ranker_base import (
    RankerBase,
    RankerCandidates,
    RankingResult,
    RemoteRankerBase,
    _build_ranking_result,
    _extract_ranking_texts,
)
from flexrag.runtime.async_client import AsyncClientMixin, ConfigT


class RankerRuntimeAdapter:
    """Direct runtime adapter for main-process raw rankers.

    The adapter constructs a raw ranker implementation and forwards calls to it
    without adding process isolation or remote runtime policies. It exists so
    ResourceManager can wrap rankers in a typed handle while keeping lifecycle
    ownership at the resource layer.
    """

    impl_cls: type[RankerBase] | None = None

    def __init__(
        self,
        config: Any,
        impl_cls: type[RankerBase] | None = None,
        **dependencies: Any,
    ) -> None:
        """Create a direct ranker runtime adapter.

        :param config: Configuration passed to the raw ranker implementation.
        :param impl_cls: Optional raw ranker implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param dependencies: Externally managed resources injected into the
            raw ranker constructor.
        :raises ValueError: If no implementation class is configured.
        """
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        self._resource = self.impl_cls(config, **dependencies)
        return

    def rank(self, query: str, candidates: RankerCandidates) -> RankingResult:
        """Rank candidates synchronously.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        return self._resource.rank(query, candidates)

    async def async_rank(
        self,
        query: str,
        candidates: RankerCandidates,
    ) -> RankingResult:
        """Rank candidates asynchronously.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        return await self._resource.async_rank(query, candidates)

    def close(self) -> None:
        """Close the wrapped ranker when it exposes a synchronous close hook."""
        close = getattr(self._resource, "close", None)
        if callable(close):
            close()
        return

    async def aclose(self) -> None:
        """Close the wrapped ranker, preferring an async close hook when present."""
        aclose = getattr(self._resource, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
            return

        close = getattr(self._resource, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        return


class RemoteRankerRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Managed runtime adapter for remote raw rankers.

    Remote raw rankers expose provider-specific asynchronous rerank primitives.
    This adapter owns lazy ranker construction, sync/async bridge behavior, and
    concurrency control for managed resource use.
    """

    impl_cls: type[RemoteRankerBase] | None = None

    def __init__(
        self,
        config: ConfigT,
        impl_cls: type[RemoteRankerBase] | None = None,
        *,
        max_concurrency: int = 1,
    ) -> None:
        """Create a remote ranker runtime adapter.

        :param config: Configuration passed to the raw ranker implementation.
        :param impl_cls: Optional raw ranker implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param max_concurrency: Maximum number of in-flight remote rank calls.
        :raises ValueError: If ``max_concurrency`` is not greater than zero.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0.")
        self._max_concurrency = max_concurrency
        return

    async def _create_client(self, config: ConfigT):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return self.impl_cls(config)

    def _get_max_concurrency(self) -> int:
        return self._max_concurrency

    async def _async_rank_core(
        self,
        query: str,
        candidates: RankerCandidates,
    ) -> RankingResult:
        client = await self._get_async_client()
        if not candidates:
            return RankingResult(query=query, candidates=[], scores=[])

        texts = _extract_ranking_texts(candidates, client.ranking_field)
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            indices, scores = await client._async_rank_batch(query, texts)
        return _build_ranking_result(
            query=query,
            candidates=candidates,
            reserve_num=client.reserve_num,
            indices=indices,
            scores=scores,
        )

    async def async_rank(
        self,
        query: str,
        candidates: RankerCandidates,
    ) -> RankingResult:
        """Rank candidates asynchronously on the managed background loop.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        return await self._run_coroutine_async(
            self._async_rank_core(query, candidates)
        )

    def rank(self, query: str, candidates: RankerCandidates) -> RankingResult:
        """Rank candidates synchronously on the managed background loop.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        return self._run_coroutine_sync(self._async_rank_core(query, candidates))
