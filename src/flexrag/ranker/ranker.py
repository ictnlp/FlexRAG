from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import Register, LOGGER_MANAGER


logger = LOGGER_MANAGER.get_logger("flexrag.rankers")


@dataclass
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


@dataclass
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
    candidates: list[RetrievedContext]
    scores: Optional[list[float]] = None


class RankerBase(ABC):
    def __init__(self, cfg: RankerBaseConfig) -> None:
        self.reserve_num = cfg.reserve_num
        self.ranking_field = cfg.ranking_field
        return

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        """Rank the candidates based on the query.

        :param query: query string.
        :param candidates: list of candidate strings.
        :type query: str
        :type candidates: list[str]
        :return: indices and scores of the ranked candidates.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        if isinstance(candidates[0], RetrievedContext):
            assert self.ranking_field is not None
            texts = [ctx.data[self.ranking_field] for ctx in candidates]
        else:
            texts = candidates
        indices, scores = self._rank(query, texts)
        if indices is None:
            assert scores is not None
            indices = np.argsort(scores)[::-1]
        if self.reserve_num > 0:
            indices = indices[: self.reserve_num]

        result = RankingResult(query=query, candidates=[])
        for idx in indices:
            result.candidates.append(candidates[idx])
        if scores is not None:
            result.scores = [scores[idx] for idx in indices]
        return result

    async def async_rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
        """The asynchronous version of `rank`."""
        if isinstance(candidates[0], RetrievedContext):
            assert self.ranking_field is not None
            texts = [ctx.data[self.ranking_field] for ctx in candidates]
        else:
            texts = candidates
        indices, scores = await self._async_rank(query, texts)
        if indices is None:
            assert scores is not None
            indices = np.argsort(scores)[::-1]
        if self.reserve_num > 0:
            indices = indices[: self.reserve_num]

        result = RankingResult(query=query, candidates=[])
        for idx in indices:
            result.candidates.append(candidates[idx])
        if scores is not None:
            result.scores = [scores[idx] for idx in indices]
        return result

    @abstractmethod
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Rank the candidates based on the query.

        :param query: query string.
        :param candidates: list of candidate strings.
        :type query: str
        :type candidates: list[str]
        :return: indices and scores of the ranked candidates.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        return

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The asynchronous version of `_rank`."""
        logger.warning("async_rank is not implemented, using the synchronous version.")
        return self._rank(query, candidates)


RANKERS = Register[RankerBase]("ranker")
