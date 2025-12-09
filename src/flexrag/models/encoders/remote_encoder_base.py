import asyncio
from abc import abstractmethod

import numpy as np

from flexrag.common import configure
from flexrag.common.async_utils import BackgroundEventLoop

from .encoder_base import EncoderBase


@configure
class RemoteEncoderBaseConfig:
    max_concurrency: int = 1


class RemoteEncoderBase(EncoderBase):
    """Base class for API-based encoders.

    The RemoteEncoderBase uses a background event loop to run async calls,
    and provides both async and sync interfaces with concurrency control.
    It manages a background event loop thread to execute asynchronous tasks and uses
    an asyncio Semaphore to limit the maximum number of concurrent API requests.

    This class provides the following public methods:
    - :meth:`encode`: Synchronous encode interface.
    - :meth:`async_encode`: Asynchronous encode interface.

    The subclasses should implement the following methods:

        >>> async def _create_client(self, config: RemoteEncoderBaseConfig):
        >>>     # Create and return the async client instance.

        >>> async def _async_encode_impl(
        >>>     self,
        >>>     client,
        >>>     texts: list[str],
        >>> ) -> np.ndarray:
        >>>     # Perform the async encode call using the client.
        >>>     ...
    """

    def __init__(self, config: RemoteEncoderBaseConfig):
        super().__init__()
        self._loop_thread = BackgroundEventLoop()
        self._semaphore = None
        self._client_lock = None
        self._client = None
        self._config = config
        return

    async def _get_async_client(self):
        """Create client lazily inside background event loop."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is None:
                self._client = await self._create_client(self._config)
        return self._client

    @abstractmethod
    async def _create_client(self, config: RemoteEncoderBaseConfig):
        """Implemented by subclasses, create and return the async client instance."""
        return

    @abstractmethod
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        """Implemented by subclasses, perform the async encode call."""
        return

    async def _async_encode_core(
        self,
        texts: list[str] | str,
        batch_size: int | None = None,
    ) -> np.ndarray:
        # Normalize input to list of strings
        if isinstance(texts, str):
            texts = [texts]

        # Batching
        if batch_size is None:
            batches = [texts]
        else:
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
        client = await self._get_async_client()

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._config.max_concurrency)

        async def _encode_task(batch: list[str]) -> np.ndarray:
            async with self._semaphore:
                return await self._async_encode_impl(client, batch)

        results = await asyncio.gather(*[_encode_task(batch) for batch in batches])

        if not results:
            return np.array([])
        return np.concatenate(results, axis=0)

    async def async_encode(self, texts: list[str] | str) -> np.ndarray:
        """Asynchronously encode the given texts into embeddings.

        :param texts: A batch of texts or a single text.
        :type texts: list[str] | str
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        thread_future = self._loop_thread.run_async(self._async_encode_core(texts))
        return await asyncio.wrap_future(thread_future)

    def encode(self, texts: list[str] | str) -> np.ndarray:
        """Encode the given texts into embeddings.

        :param texts: A batch of texts or a single text.
        :type texts: list[str] | str
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        future = self._loop_thread.run_async(self._async_encode_core(texts))
        return future.result()

    def encode_batch(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        log_interval: int = 1000,
    ) -> np.ndarray:
        """Encode the given texts into embeddings in batches.

        :param texts: A batch of texts.
        :type texts: list[str] | str
        :param batch_size: The size of each batch. Defaults to 32.
        :type batch_size: int
        :param log_interval: The interval for logging progress. Defaults to 1000.
            If set to 0, no logs will be shown.
            Note that logging is not used in this method, so this parameter is currently unused.
        :type log_interval: int
        :return: A batch of embeddings.
        :rtype: np.ndarray
        """
        future = self._loop_thread.run_async(
            self._async_encode_core(texts, batch_size=batch_size)
        )
        return future.result()
