import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias

import numpy as np

from flexrag.common import Register

PairScorerInput: TypeAlias = tuple[str, str] | list[tuple[str, str]]


def _normalize_score_pairs(pairs: PairScorerInput) -> list[tuple[str, str]]:
    def _validate_pair(pair) -> tuple[str, str]:
        if not (
            isinstance(pair, tuple)
            and len(pair) == 2
            and isinstance(pair[0], str)
            and isinstance(pair[1], str)
        ):
            raise TypeError("Scorer pairs must be tuple[str, str].")
        return pair

    if isinstance(pairs, tuple):
        return [_validate_pair(pairs)]
    return [_validate_pair(pair) for pair in pairs]


class PairScorerProtocol(Protocol):
    """Protocol for directly usable raw pair scorers.

    Raw pair scorers expose a common canonical-batch interface for direct use.
    Implementations do not provide runtime policies such as deployment
    batching, progress logging, or process isolation.
    """

    def score(
        self,
        pairs: PairScorerInput,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score one query-candidate pair or a batch of pairs.

        :param pairs: Query-candidate pair or pairs to score.
        :param batch_size: Optional per-call batch size override.
        :return: One score for each input pair.
        """
        ...

    async def async_score(
        self,
        pairs: PairScorerInput,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score one query-candidate pair or a batch asynchronously.

        :param pairs: Query-candidate pair or pairs to score.
        :param batch_size: Optional per-call batch size override.
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

    def score(
        self,
        pairs: PairScorerInput,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score one query-candidate pair or a batch of pairs.

        :param pairs: Query-candidate pair or pairs to score.
        :param batch_size: Optional per-call batch size override.
        :return: One score for each input pair.
        """
        normalized_pairs = _normalize_score_pairs(pairs)
        if not normalized_pairs:
            return np.array([])
        resolved_batch_size = batch_size or self.batch_size
        results = [
            self._score_batch(normalized_pairs[i : i + resolved_batch_size])
            for i in range(0, len(normalized_pairs), resolved_batch_size)
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

    async def async_score(
        self,
        pairs: PairScorerInput,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Score one query-candidate pair or a batch asynchronously.

        :param pairs: Query-candidate pair or pairs to score.
        :param batch_size: Optional per-call batch size override.
        :return: One score for each input pair.
        """
        return await asyncio.to_thread(self.score, pairs, batch_size=batch_size)


SCORERS = Register[PairScorerProtocol]("scorer")
