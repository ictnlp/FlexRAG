import asyncio
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from flexrag.common import Register


class PairScorerProtocol(Protocol):
    """Protocol for directly usable raw pair scorers.

    Raw pair scorers expose a common canonical-batch interface for direct use.
    Implementations do not provide runtime policies such as deployment
    batching, progress logging, or process isolation.
    """

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score a batch of query-candidate pairs.

        :param pairs: Query-candidate pairs to score.
        :return: One score for each input pair.
        """
        ...

    async def async_score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score a batch of query-candidate pairs asynchronously.

        :param pairs: Query-candidate pairs to score.
        :return: One score for each input pair.
        """
        ...


class LocalPairScorerBase(ABC):
    """Thin base class for directly usable local pair scorers.

    Subclasses implement synchronous canonical-batch ``_score_batch``. The
    public ``score`` method splits direct-use calls according to ``batch_size``
    and merges the resulting arrays. The async method is a convenience wrapper
    built with ``asyncio.to_thread``; it keeps an event loop responsive but does
    not provide process isolation, progress logging, or true Python-level
    parallelism.
    """

    def __init__(self, batch_size: int = 32) -> None:
        """Initialize direct-use local scorer batching.

        :param batch_size: Maximum batch size used by the raw local scorer's
            public ``score`` method.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.batch_size = batch_size
        return

    def score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score a batch of query-candidate pairs.

        :param pairs: Query-candidate pairs to score.
        :return: One score for each input pair.
        """
        if not pairs:
            return np.array([])
        results = [
            self._score_batch(pairs[i : i + self.batch_size])
            for i in range(0, len(pairs), self.batch_size)
        ]
        if len(results) == 1:
            return results[0]
        return np.concatenate(results, axis=0)

    @abstractmethod
    def _score_batch(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score one implementation batch.

        :param pairs: Query-candidate pairs to score.
        :return: One score for each input pair.
        """
        return

    async def async_score(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """Score a batch of query-candidate pairs asynchronously.

        :param pairs: Query-candidate pairs to score.
        :return: One score for each input pair.
        """
        return await asyncio.to_thread(self.score, pairs)


SCORERS = Register[PairScorerProtocol]("scorer")
