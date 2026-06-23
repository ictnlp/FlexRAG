import asyncio
from typing import Any

import numpy as np

from flexrag.common import ContentPart, ProgressDisplay, SimpleProgressLogger
from flexrag.models.encoders.encoder_base import (
    EncoderInputs,
    _normalize_encoder_inputs,
)

from .common import split_batches, unwrap_exception_group


def _merge_encode_results(results: list[None | np.ndarray]) -> np.ndarray:
    if not results:
        return np.array([])
    ready_results = [result for result in results if result is not None]
    if len(ready_results) != len(results):
        raise RuntimeError("Some encode tasks did not produce results.")
    if len(ready_results) == 1:
        return ready_results[0]
    return np.concatenate(ready_results, axis=0)


class EncoderInvocation:
    """Invocation semantics for managed encoder resources.

    The invocation owns encoder input normalization, deployment batch splitting,
    progress reporting, embedding result merging, and ``embedding_size`` access.
    The runtime only executes primitive calls.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        batch_method: str,
        batch_size: int = 32,
    ) -> None:
        """Create an encoder invocation.

        :param runtime: Runtime adapter used to execute primitive calls.
        :param batch_method: Runtime method used for one canonical content batch.
        :param batch_size: Deployment batch size.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.runtime = runtime
        self._batch_method = batch_method
        self._batch_size = batch_size
        return

    async def _async_encode_core(
        self,
        inputs: list[ContentPart],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        batches = split_batches(inputs, self._batch_size)
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(inputs), interval=log_interval, display=display
        ) as p_logger:

            async def _encode_task(idx: int, batch: list[ContentPart]) -> None:
                results[idx] = await self.runtime.acall(self._batch_method, batch)
                p_logger.update(len(batch), desc="Encoding")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _encode_task(idx, batch),
                            name=f"encode_batch_{idx}",
                        )
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        return _merge_encode_results(results)

    async def async_encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Encode inputs asynchronously.

        :param inputs: Input item or batch to encode.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Encoded embeddings.
        """
        normalized_inputs = _normalize_encoder_inputs(inputs)
        return await self.runtime.run_async(
            self._async_encode_core(
                normalized_inputs,
                log_interval=log_interval,
                display=display,
            )
        )

    def encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Encode inputs synchronously.

        :param inputs: Input item or batch to encode.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Encoded embeddings.
        """
        normalized_inputs = _normalize_encoder_inputs(inputs)
        return self.runtime.run_sync(
            self._async_encode_core(
                normalized_inputs,
                log_interval=log_interval,
                display=display,
            )
        )

    async def _async_embedding_size(self) -> int | None:
        try:
            return await self.runtime.agetattr("embedding_size")
        except AttributeError:
            return None

    @property
    def embedding_size(self) -> int | None:
        return self.runtime.run_sync(self._async_embedding_size())
