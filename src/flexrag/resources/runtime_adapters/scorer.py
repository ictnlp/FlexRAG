import asyncio
from abc import abstractmethod

import numpy as np

from flexrag.common import ProgressDisplay, SimpleProgressLogger
from flexrag.models.scorers.scorer_base import (
    LocalPairScorerBase,
    PairScorerInput,
    _normalize_score_pairs,
)
from flexrag.runtime.async_client import AsyncClientMixin, ConfigT
from flexrag.runtime.process_worker_pool import ProcessWorkerPoolClient


def _merge_score_results(results: list[None | np.ndarray]) -> np.ndarray:
    if not results:
        return np.array([])
    ready_results = [result for result in results if result is not None]
    if len(ready_results) != len(results):
        raise RuntimeError("Some score tasks did not produce results.")
    if len(ready_results) == 1:
        return ready_results[0]
    return np.concatenate(ready_results, axis=0)


class ScorerRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Managed call adapter for pair scorer-like resources.

    This base adapter provides the public sync and async call surface for pair
    scorer resources. It submits work to the managed background event loop,
    reports progress, and delegates actual scoring work to subclass core
    methods.
    """

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @abstractmethod
    async def _async_score_core(
        self,
        pairs: list[tuple[str, str]],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return

    async def async_score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Score query-candidate pairs asynchronously.

        :param pairs: Query-candidate pair or pairs to score.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: One score for each input pair.
        """
        normalized_pairs = _normalize_score_pairs(pairs)
        return await self._run_coroutine_async(
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
        """Score query-candidate pairs synchronously.

        :param pairs: Query-candidate pair or pairs to score.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: One score for each input pair.
        """
        normalized_pairs = _normalize_score_pairs(pairs)
        return self._run_coroutine_sync(
            self._async_score_core(
                normalized_pairs,
                log_interval=log_interval,
                display=display,
            )
        )


class ProcessScorerAdapter(ScorerRuntimeAdapter):
    """Process-backed runtime adapter for local raw pair scorers.

    Local raw scorers process canonical pair batches synchronously. This
    adapter creates a process worker pool lazily, splits managed calls into
    deployment batches, dispatches batches across available workers, and merges
    results back into input order.
    """

    impl_cls: type[LocalPairScorerBase] | None = None

    def __init__(
        self,
        config,
        impl_cls: type[LocalPairScorerBase] | None = None,
        *,
        batch_size: int = 32,
        device_groups: list[list[int]] | None = None,
    ):
        """Create a process-backed scorer runtime adapter.

        :param config: Configuration passed to the raw scorer implementation.
        :param impl_cls: Optional raw scorer implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param batch_size: Deployment batch size used for worker RPC calls.
        :param device_groups: Worker device placement. ``None`` creates one
            worker inheriting the current environment, ``[]`` creates one
            CPU-only worker, and non-empty groups create one worker per group.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self._batch_size = batch_size
        self._device_groups = device_groups
        self._worker_count = 1
        return

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        client = ProcessWorkerPoolClient.from_device_groups(
            self.impl_cls,
            config,
            self._device_groups,
        )
        self._worker_count = len(client)
        return client

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return self._worker_count

    async def _async_score_core(
        self,
        pairs: list[tuple[str, str]],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        batches = [
            pairs[i : i + self._batch_size]
            for i in range(0, len(pairs), self._batch_size)
        ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(pairs), interval=log_interval, display=display
        ) as p_logger:

            async def _score_task(idx: int, batch: list[tuple[str, str]]) -> None:
                async with semaphore:
                    res = await client.call_available("_score_batch", batch)
                results[idx] = res
                p_logger.update(len(batch), desc="Scoring")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _score_task(idx, batch), name=f"score_batch_{idx}"
                        )
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        return _merge_score_results(results)
