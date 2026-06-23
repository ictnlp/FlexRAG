import asyncio
from typing import Any

import numpy as np

from flexrag.common import ProgressDisplay, SimpleProgressLogger
from flexrag.models.scorers.scorer_base import (
    PairScorerInput,
    _normalize_score_pairs,
)

from .common import split_batches, unwrap_exception_group


def _merge_score_results(results: list[None | np.ndarray]) -> np.ndarray:
    if not results:
        return np.array([])
    ready_results = [result for result in results if result is not None]
    if len(ready_results) != len(results):
        raise RuntimeError("Some score tasks did not produce results.")
    if len(ready_results) == 1:
        return ready_results[0]
    return np.concatenate(ready_results, axis=0)


class ScorerInvocation:
    """Invocation semantics for managed pair scorer resources."""

    def __init__(
        self,
        runtime: Any,
        *,
        score_method: str,
        batch_size: int = 32,
    ) -> None:
        """Create a scorer invocation.

        :param runtime: Runtime adapter used to execute primitive calls.
        :param score_method: Primitive method for one pair batch.
        :param batch_size: Deployment batch size.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.runtime = runtime
        self._score_method = score_method
        self._batch_size = batch_size
        return

    async def _async_score_core(
        self,
        pairs: list[tuple[str, str]],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        batches = split_batches(pairs, self._batch_size)
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(pairs), interval=log_interval, display=display
        ) as p_logger:

            async def _score_task(idx: int, batch: list[tuple[str, str]]) -> None:
                results[idx] = await self.runtime.acall(self._score_method, batch)
                p_logger.update(len(batch), desc="Scoring")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _score_task(idx, batch),
                            name=f"score_batch_{idx}",
                        )
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        return _merge_score_results(results)

    async def async_score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Score query-candidate pairs asynchronously."""
        normalized_pairs = _normalize_score_pairs(pairs)
        return await self.runtime.run_async(
            self._async_score_core(
                normalized_pairs,
                log_interval=log_interval,
                display=display,
            )
        )

    def score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Score query-candidate pairs synchronously."""
        normalized_pairs = _normalize_score_pairs(pairs)
        return self.runtime.run_sync(
            self._async_score_core(
                normalized_pairs,
                log_interval=log_interval,
                display=display,
            )
        )
