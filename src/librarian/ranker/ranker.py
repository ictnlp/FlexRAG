from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from librarian.retriever import RetrievedContext
from librarian.utils import Register, LOGGER_MANAGER


logger = LOGGER_MANAGER.get_logger("librarian.rankers")


@dataclass
class RankerConfig:
    reserve_num: int = -1
    ranking_field: Optional[str] = None


@dataclass
class RankingResult:
    query: str
    candidates: list[RetrievedContext]
    scores: Optional[list[float]] = None


class RankerBase(ABC):
    def __init__(self, cfg: RankerConfig) -> None:
        self.reserve_num = cfg.reserve_num
        self.ranking_field = cfg.ranking_field
        return

    def rank(
        self, query: str, candidates: list[RetrievedContext | str]
    ) -> RankingResult:
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

        Args:
            query (str): query string.
            candidates (list[str]): list of candidate strings.

        Returns:
            tuple[np.ndarray, np.ndarray]: indices and scores of the ranked candidates.
        """
        return

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """The asynchronous version of `_rank`."""
        logger.warning("async_rank is not implemented, using the synchronous version.")
        return self._rank(query, candidates)


RANKERS = Register[RankerBase]("ranker")
