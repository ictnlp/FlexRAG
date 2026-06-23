import asyncio
import inspect
from typing import Any

from flexrag.runtime.async_client import AsyncClientMixin, ConfigT


class RemoteRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Remote runtime for asynchronous raw resources.

    The runtime owns lazy raw-client construction, background-loop execution,
    concurrency limits, requests-per-minute limiting, retry, and lifecycle.
    Interface-specific input handling, progress, and result merging belong to
    invocation objects.
    """

    impl_cls: type[Any] | None = None

    def __init__(
        self,
        config: ConfigT,
        impl_cls: type[Any] | None = None,
        *,
        max_concurrency: int = 1,
        rpm: float = 0,
        retry_times: int = 0,
        retry_min_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        """Create a remote runtime.

        :param config: Configuration passed to the raw remote implementation.
        :param impl_cls: Optional raw remote implementation class. When omitted,
            subclasses must set ``impl_cls``.
        :param max_concurrency: Maximum number of in-flight remote calls.
        :param rpm: Requests-per-minute limit. ``0`` disables rate limiting.
        :param retry_times: Number of retries after the initial attempt. ``0``
            disables retry.
        :param retry_min_delay: Initial retry delay in seconds.
        :param retry_max_delay: Maximum retry delay in seconds.
        :raises ValueError: If any runtime policy value is invalid.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0.")
        if rpm < 0:
            raise ValueError("rpm must be non-negative.")
        if retry_times < 0:
            raise ValueError("retry_times must be non-negative.")
        if retry_min_delay < 0:
            raise ValueError("retry_min_delay must be non-negative.")
        if retry_max_delay < retry_min_delay:
            raise ValueError(
                "retry_max_delay must be greater than or equal to retry_min_delay."
            )
        self._max_concurrency = max_concurrency
        self._rpm = rpm
        self._retry_times = retry_times
        self._retry_min_delay = retry_min_delay
        self._retry_max_delay = retry_max_delay
        self._rpm_lock: asyncio.Lock | None = None
        self._next_request_time: float = 0.0
        return

    async def _create_client(self, config: ConfigT) -> Any:
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return self.impl_cls(config)

    def _get_max_concurrency(self) -> int:
        return self._max_concurrency

    def run_sync(self, coro):
        """Run a coroutine on the managed runtime loop synchronously."""
        return self._run_coroutine_sync(coro)

    async def run_async(self, coro):
        """Run a coroutine on the managed runtime loop asynchronously."""
        return await self._run_coroutine_async(coro)

    async def _wait_for_rpm(self) -> None:
        if self._rpm <= 0:
            return
        if self._rpm_lock is None:
            self._rpm_lock = asyncio.Lock()
        interval = 60.0 / self._rpm
        async with self._rpm_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_request_time > now:
                await asyncio.sleep(self._next_request_time - now)
                now = loop.time()
            self._next_request_time = max(self._next_request_time, now) + interval
        return

    def _retry_delay(self, retry_idx: int) -> float:
        delay = self._retry_min_delay * (2**retry_idx)
        return min(delay, self._retry_max_delay)

    async def _run_with_retry(self, call, *args: Any, **kwargs: Any) -> Any:
        for attempt in range(self._retry_times + 1):
            await self._wait_for_rpm()
            try:
                result = call(*args, **kwargs)
                if inspect.isawaitable(result):
                    return await result
                return result
            except Exception:
                if attempt >= self._retry_times:
                    raise
                delay = self._retry_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)
        raise RuntimeError("Remote runtime retry loop exited unexpectedly.")

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the lazy remote raw resource.

        :param method: Raw resource method name.
        :param args: Positional arguments forwarded to the method.
        :param kwargs: Keyword arguments forwarded to the method.
        :return: Method return value.
        """
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            return await self._run_with_retry(
                getattr(client, method),
                *args,
                **kwargs,
            )

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously call a method on the lazy remote raw resource."""
        return self.run_sync(self.acall(method, *args, **kwargs))

    async def agetattr(self, name: str) -> Any:
        """Return an attribute from the lazy remote raw resource."""
        client = await self._get_async_client()
        return getattr(client, name)

    def getattr(self, name: str) -> Any:
        """Synchronously return an attribute from the lazy remote raw resource."""
        return self.run_sync(self.agetattr(name))
