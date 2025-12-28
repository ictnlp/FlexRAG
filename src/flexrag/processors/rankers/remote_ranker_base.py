import asyncio
from abc import abstractmethod

import numpy as np

from flexrag.common import configure
from flexrag.common.async_utils import BackgroundEventLoop
from flexrag.common.dataclasses import RetrievedContext

from .ranker_base import RankerBase, RankerBaseConfig, RankingResult


@configure
class RemoteRankerBaseConfig(RankerBaseConfig):
    max_concurrency: int = 1


class RemoteRankerBase(RankerBase):
    """Base class for API-based rankers.

    The RemoteRankerBase uses a background event loop to run async calls,
    and provides both async and sync interfaces with concurrency control.
    It manages a background event loop thread to execute asynchronous tasks and uses
    an asyncio Semaphore to limit the maximum number of concurrent API requests.

    This class provides the following public methods:
    - :meth:`rank`: Synchronous rank interface.
    - :meth:`async_rank`: Asynchronous rank interface.

    The subclasses should implement the following methods:

        >>> async def _create_client(self, config: RemoteRankerBaseConfig):
        >>>     # Create and return the async client instance.

        >>> async def _async_rank_impl(
        >>>     self,
        >>>     client,
        >>>     query: str,
        >>>     candidates: list[str],
        >>> ) -> tuple[np.ndarray, np.ndarray | None]:
        >>>     # Perform the async rank call using the client.
        >>>     # Return indices and scores of the ranked candidates.
        >>>     ...
    """

    def __init__(self, config: RemoteRankerBaseConfig):
        super().__init__(config)
        self._loop_thread = BackgroundEventLoop()
        self._semaphore = None
        self._client_lock = None
        self._client = None
        self._config = config
        return

    async def _get_async_client(self):
        """Create client lazily inside background event loop."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is None:
                self._client = await self._create_client(self._config)
        return self._client

    @abstractmethod
    async def _create_client(self, config: RemoteRankerBaseConfig):
        """Implemented by subclasses, create and return the async client instance."""
        return

    @abstractmethod
    async def _async_rank_impl(
        self, client, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Implemented by subclasses, perform the async rank call.

        :return: indices and scores of the ranked candidates.
            If the scores are provided, the ranker will sort by scores.
            If the scores are None, the ranker will use the indices returned by the API.
        :rtype: tuple[np.ndarray, np.ndarray | None]
        """
        return

    async def _async_rank_core(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        # prepare texts
        texts = []
        for cand in candidates:
            if isinstance(cand, str):
                texts.append(cand)
            else:
                if self.ranking_field is None:
                    raise ValueError(
                        "ranking_field must be specified when ranking RetrievedContext"
                    )
                texts.append(cand.data[self.ranking_field])

        # get client
        client = await self._get_async_client()

        # rank with concurrency control
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
        async with self._semaphore:
            indices, scores = await self._async_rank_impl(client, query, texts)

        # if scores are provided, ignore indices and sort by scores
        if scores is not None:
            scores = np.array(scores)
            indices = np.argsort(scores)[::-1]

        # reserve
        if self.reserve_num > 0:
            indices = indices[: self.reserve_num]

        sorted_candidates = [candidates[i] for i in indices]
        if scores is not None:
            sorted_scores = scores[indices].tolist()
        else:
            sorted_scores = None

        return RankingResult(
            query=query,
            candidates=sorted_candidates,
            scores=sorted_scores,
        )

    async def async_rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        """Asynchronously rank the candidates based on the query.

        :param query: query string.
        :param candidates: list of candidate strings or RetrievedContext.
        :type query: str
        :type candidates: list[str | RetrievedContext]
        :return: RankingResult containing ranked candidates and scores.
        :rtype: RankingResult
        """
        thread_future = self._loop_thread.run_async(
            self._async_rank_core(query, candidates)
        )
        return await asyncio.wrap_future(thread_future)

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        future = self._loop_thread.run_async(self._async_rank_core(query, candidates))
        return future.result()
