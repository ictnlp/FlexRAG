import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Protocol

import numpy as np

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.common.dataclasses import RetrievedContext

logger = LOGGER_MANAGER.get_logger("flexrag.rankers")


@configure
class RankerBaseConfig:
    """The configuration for the ranker.

    :param reserve_num: the number of candidates to reserve.
        If it is less than 0, all candidates will be reserved. Default is -1.
    :type reserve_num: int
    :param ranking_field: the field name of the ranking field in the retrieved context.
        If it is None, the ranker will only accept a list of strings as candidates.
    :type ranking_field: Optional[str]
    """

    reserve_num: int = -1
    ranking_field: Optional[str] = None


@configure
class RankingResult:
    """The result of ranking.

    :param query: the query string. Required.
    :type query: str
    :param candidates: the ranked candidates.
        The results are sorted in descending order by relevance. Required.
    :type candidates: list[RetrievedContext | str]
    :param scores: the scores of the ranked candidates. Optional.
    :type scores: Optional[list[float]]
    """

    query: str
    candidates: list[RetrievedContext | str]
    scores: Optional[list[float]] = None


class RankerProtocol(Protocol):
    """Structural interface shared by raw rankers and managed ranker handles."""

    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank candidates based on a query.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        ...

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank candidates asynchronously based on a query.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        ...


def _extract_ranking_texts(
    candidates: list[RetrievedContext | str],
    ranking_field: str | None,
) -> list[str]:
    texts: list[str] = []
    for candidate in candidates:
        if isinstance(candidate, str):
            texts.append(candidate)
            continue
        if ranking_field is None:
            raise ValueError(
                "ranking_field must be specified when ranking RetrievedContext"
            )
        texts.append(candidate.data[ranking_field])
    return texts


def _build_ranking_result(
    query: str,
    candidates: list[RetrievedContext | str],
    *,
    reserve_num: int,
    indices: np.ndarray | None,
    scores: np.ndarray | None,
) -> RankingResult:
    if scores is not None:
        scores = np.asarray(scores)
        indices = np.argsort(scores)[::-1]
    elif indices is None:
        raise ValueError("Either indices or scores must be provided.")
    else:
        indices = np.asarray(indices)

    if reserve_num > 0:
        indices = indices[:reserve_num]

    ranked_candidates = [candidates[int(idx)] for idx in indices]
    ranked_scores = scores[indices].tolist() if scores is not None else None
    return RankingResult(
        query=query,
        candidates=ranked_candidates,
        scores=ranked_scores,
    )


class RankerBase(ABC):
    """Base class for rankers.
    The ranker can rank candidates based on a query.
    The subclasses must implement the `rank` method.
    """

    def __init__(self, cfg: RankerBaseConfig) -> None:
        self.reserve_num = cfg.reserve_num
        self.ranking_field = cfg.ranking_field
        return

    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank the candidates based on the query.

        :param query: query string.
        :type query: str
        :param candidates: list of candidate strings.
        :type candidates: list[RetrievedContext | str]
        :return: RankingResult containing ranked candidates and scores.
        :rtype: RankingResult
        """
        return

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """The asynchronous version of `rank`."""
        logger.warning(
            "Current ranker does not support asynchronous rank,"
            "thus the code will be run in synchronous mode"
        )
        return self.rank(query, candidates)


class RemoteRankerBase(RankerBase):
    """Thin base class for directly usable remote rankers.

    Subclasses implement the provider-specific asynchronous rerank primitive
    over a canonical text batch. The public methods handle direct-use candidate
    extraction and result construction, but they do not provide runtime
    policies such as background-loop execution or concurrency control.
    """

    @staticmethod
    def _ensure_sync_bridge_allowed(method_name: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            f"{method_name} cannot be called from a running event loop. "
            f"Use async_{method_name} instead."
        )

    @abstractmethod
    async def _async_rank_batch(
        self,
        query: str,
        candidates: list[str],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Rank one canonical candidate text batch asynchronously.

        :param query: Query string.
        :param candidates: Candidate texts to rank.
        :return: Ranked indices and optional scores. If scores are provided,
            they are used to derive final ranking order.
        """
        return

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank candidates asynchronously for direct raw-ranker use.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        """
        if not candidates:
            return RankingResult(query=query, candidates=[], scores=[])
        texts = _extract_ranking_texts(candidates, self.ranking_field)
        indices, scores = await self._async_rank_batch(query, texts)
        return _build_ranking_result(
            query=query,
            candidates=candidates,
            reserve_num=self.reserve_num,
            indices=indices,
            scores=scores,
        )

    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        """Rank candidates synchronously for direct raw-ranker use.

        :param query: Query string.
        :param candidates: Candidate strings or retrieved contexts.
        :return: Ranked candidates and optional scores.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("rank")
        return asyncio.run(self.async_rank(query, candidates))
