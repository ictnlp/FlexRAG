import asyncio
from abc import abstractmethod

import numpy as np

from flexrag.common.logging import SimpleProgressLogger
from flexrag.models.async_client_base import AsyncClientMixin, ConfigT

from .scorer_base import PairScorerBase


class AsyncScorerBase(PairScorerBase, AsyncClientMixin[ConfigT]):
    """Base class for scorer proxies backed by an async client/runtime."""

    def __init__(self, config: ConfigT):
        AsyncClientMixin.__init__(self, config)
        return

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @abstractmethod
    async def _async_score_impl(
        self, client, pairs: list[tuple[str, str]]
    ) -> np.ndarray:
        return

    async def _async_score_core(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int | None = None,
        log_interval: int = 1000,
    ) -> np.ndarray:
        if batch_size is None:
            batches = [pairs]
        else:
            batches = [
                pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)
            ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        p_logger = SimpleProgressLogger(total=len(pairs), interval=log_interval)
        results: list[None | np.ndarray] = [None] * len(batches)

        async def _score_task(idx: int, batch: list[tuple[str, str]]) -> None:
            async with semaphore:
                res = await self._async_score_impl(client, batch)
            results[idx] = res
            p_logger.update(len(batch), desc="Scoring")
            return

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, batch in enumerate(batches):
                    tg.create_task(_score_task(idx, batch), name=f"score_batch_{idx}")
        except ExceptionGroup as exc:
            raise self._unwrap_exception_group(exc) from exc

        if not results:
            return np.array([])
        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some score tasks did not produce results.")
        if len(ready_results) == 1:
            return ready_results[0]
        return np.concatenate(ready_results, axis=0)

    async def async_score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        return await self._run_coroutine_async(
            self._async_score_core(
                pairs,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    def score(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        return self._run_coroutine_sync(
            self._async_score_core(
                pairs,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )
