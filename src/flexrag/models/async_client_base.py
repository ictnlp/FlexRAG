import asyncio
import atexit
import threading
from abc import abstractmethod
from typing import Generic, TypeVar

from flexrag.common.async_utils import BackgroundEventLoop

ConfigT = TypeVar("ConfigT")


class AsyncClientMixin(Generic[ConfigT]):
    """Shared lifecycle and sync/async bridge for client-backed components.

    This mixin centralizes the mechanics needed by components whose real work is
    performed by an asynchronously created client, such as a remote API client
    or a local process-backed runtime.

    The mixin works in four steps:

    1. ``__init__`` stores the config and creates a dedicated
       :class:`BackgroundEventLoop`. Synchronous public methods can submit
       coroutines to this loop and wait for their results without requiring the
       caller to manage an event loop.
    2. ``_get_async_client`` lazily creates the underlying client the first time
       it is needed. Creation is guarded by an ``asyncio.Lock`` so concurrent
       requests do not race and instantiate the client more than once.
    3. ``_get_async_semaphore`` lazily creates a semaphore from the concurrency
       limit returned by ``_get_max_concurrency``. Subclasses can override that
       hook to choose a component-specific default.
    4. ``close`` shuts down the client and the background event loop exactly
       once. It also unregisters the ``atexit`` hook so the object can be
       cleaned up explicitly in long-running processes and still be cleaned up
       safely at interpreter exit.

    Subclasses are expected to implement ``_create_client`` and may optionally
    override ``_close_client`` when the underlying client requires explicit
    asynchronous cleanup.
    """

    def __init__(self, config: ConfigT):
        self._loop_thread = BackgroundEventLoop()
        self._client = None
        self._client_lock = None
        self._semaphore = None
        self._config = config
        self._close_lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)
        return

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed.")
        return

    async def _get_async_client(self):
        self._ensure_not_closed()
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        async with self._client_lock:
            if self._client is None:
                self._client = await self._create_client(self._config)
        return self._client

    def _get_max_concurrency(self) -> int:
        return 1

    async def _get_async_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._get_max_concurrency())
        return self._semaphore

    def _run_coroutine_sync(self, coro):
        self._ensure_not_closed()
        future = self._loop_thread.run_async(coro)
        return future.result()

    async def _run_coroutine_async(self, coro):
        self._ensure_not_closed()
        future = self._loop_thread.run_async(coro)
        return await asyncio.wrap_future(future)

    @abstractmethod
    async def _create_client(self, config: ConfigT):
        return

    async def _close_client(self, client) -> None:
        return

    async def _aclose(self) -> None:
        if self._client is None:
            return
        try:
            await self._close_client(self._client)
        finally:
            self._client = None
            self._client_lock = None
            self._semaphore = None
        return

    def close(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        if self._client is not None:
            try:
                self._loop_thread.run_async(self._aclose()).result()
            except Exception:
                pass
        self._loop_thread.stop()
        return

    def __enter__(self):
        self._ensure_not_closed()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return
