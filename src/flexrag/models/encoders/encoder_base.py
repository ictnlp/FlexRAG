import asyncio
from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from flexrag.common import Register, SimpleProgressLogger
from flexrag.models.async_client_base import AsyncClientMixin, ConfigT


class EncoderProtocol(Protocol):
    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray: ...

    async def async_encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray: ...

    @property
    def embedding_size(self) -> int | None: ...


class EncoderBase(AsyncClientMixin[ConfigT], ABC):
    """Base class for client-backed encoder implementations."""

    def __init__(self, config: ConfigT):
        AsyncClientMixin.__init__(self, config)
        return

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @abstractmethod
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        return

    async def _async_encode_core(
        self,
        texts: list[str] | str,
        batch_size: int | None = None,
        log_interval: int = 1000,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        if batch_size is None:
            batches = [texts]
        else:
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        p_logger = SimpleProgressLogger(total=len(texts), interval=log_interval)

        results: list[None | np.ndarray] = [None] * len(batches)

        async def _encode_task(idx: int, batch: list[str]) -> None:
            async with semaphore:
                res = await self._async_encode_impl(client, batch)
            results[idx] = res
            p_logger.update(len(batch), desc="Encoding")
            return

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, batch in enumerate(batches):
                    tg.create_task(_encode_task(idx, batch), name=f"encode_batch_{idx}")
        except ExceptionGroup as exc:
            raise self._unwrap_exception_group(exc) from exc

        if not results:
            return np.array([])
        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some encode tasks did not produce results.")
        if len(ready_results) == 1:
            return ready_results[0]
        return np.concatenate(ready_results, axis=0)

    async def async_encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        return await self._run_coroutine_async(
            self._async_encode_core(
                texts,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        return self._run_coroutine_sync(
            self._async_encode_core(
                texts,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        return


ENCODERS = Register[EncoderProtocol]("encoder")
