from abc import abstractmethod

import numpy as np

from flexrag.common import configure
from flexrag.common.dataclasses import RetrievedContext
from flexrag.models.async_client_base import AsyncClientMixin

from .ranker_base import RankerBase, RankerBaseConfig, RankingResult


@configure
class RemoteRankerBaseConfig(RankerBaseConfig):
    max_concurrency: int = 1


class RemoteRankerBase(
    AsyncClientMixin[RemoteRankerBaseConfig],
    RankerBase,
):
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
        RankerBase.__init__(self, config)
        AsyncClientMixin.__init__(self, config)
        return

    def _get_max_concurrency(self) -> int:
        return max(1, self._config.max_concurrency)

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
        semaphore = await self._get_async_semaphore()
        async with semaphore:
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
        return await self._run_coroutine_async(self._async_rank_core(query, candidates))

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        return self._run_coroutine_sync(self._async_rank_core(query, candidates))
