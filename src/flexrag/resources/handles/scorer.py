from __future__ import annotations

import numpy as np

from flexrag.models.scorers.scorer_base import PairScorerInput, _normalize_score_pairs

from ..runtime import RuntimeCall
from .base import TypedHandle


class ScorerHandle(TypedHandle):
    """Typed proxy for pair scorer resources.

    The handle normalizes score pairs, splits them using the target batch size,
    submits primitive runtime calls, and merges score arrays. It does not own
    the scorer lifecycle.
    """

    def score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> np.ndarray:
        """Synchronously score query-document pairs.

        :param pairs: Pair input accepted by the formal scorer API.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Score array in input order.
        """
        normalized = _normalize_score_pairs(pairs)
        if not normalized:
            return np.array([])

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "score",
                args=(batch,),
                kwargs={"batch_size": batch_size},
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results = self._target.batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Scoring",
        )
        return self._merge_arrays(results)

    async def async_score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> np.ndarray:
        """Asynchronously score query-document pairs.

        :param pairs: Pair input accepted by the formal scorer API.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Score array in input order.
        """
        normalized = _normalize_score_pairs(pairs)
        if not normalized:
            return np.array([])

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "async_score",
                args=(batch,),
                kwargs={"batch_size": batch_size},
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results = await self._target.async_batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Scoring",
        )
        return self._merge_arrays(results)

    @staticmethod
    def _merge_arrays(results: list[np.ndarray]) -> np.ndarray:
        if not results:
            return np.array([])
        arrays = [np.asarray(result) for result in results]
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays, axis=0)
