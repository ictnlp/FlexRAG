from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .base import RuntimeCall
from .scheduler import RateLimiter, RetryTimeoutPolicy
from .target_base import RuntimeTargetBase


class AsyncTarget(RuntimeTargetBase):
    """Runtime target for async-first client-backed resources.

    ``AsyncTarget`` lazily constructs the raw resource on first use and injects
    refs as real typed handles in the current process. Synchronous handle calls
    prefer a raw ``async_<method>`` twin when it exists, which avoids invoking
    raw sync bridges inside the target event loop. The target also owns
    retry/backoff, timeout, RPM limiting, and raw resource shutdown.
    """

    def __init__(
        self,
        raw_cls: type[Any],
        config: Any,
        refs: dict[str, Any] | None = None,
        *,
        batch_size: int = 1,
        max_concurrency: int = 1,
        rpm: float = 0,
        retry_times: int = 0,
        retry_min_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        timeout: float = 0,
    ) -> None:
        """Create an async-first target with lazy raw resource construction.

        :param raw_cls: Raw resource class to instantiate lazily.
        :param config: Config object passed as the first constructor argument.
        :param refs: Constructor refs materialized as typed handles.
        :param batch_size: Public-call batch size exposed to handles.
        :param max_concurrency: Maximum primitive calls to run concurrently.
        :param rpm: Attempt-level request-per-minute limit. ``0`` disables
            rate limiting.
        :param retry_times: Number of retries after the initial attempt.
        :param retry_min_delay: Minimum retry backoff delay in seconds.
        :param retry_max_delay: Maximum retry backoff delay in seconds.
        :param timeout: Per-attempt timeout in seconds. ``0`` disables timeout.
        """
        self._raw_cls = raw_cls
        self._config = config
        self._refs = refs or {}
        self._raw: Any | None = None
        self._raw_lock: asyncio.Lock | None = None
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            call_policy=RetryTimeoutPolicy(
                RateLimiter(rpm=rpm),
                retry_times=retry_times,
                retry_min_delay=retry_min_delay,
                retry_max_delay=retry_max_delay,
                timeout=timeout,
            ),
        )
        return

    async def _get_raw(self) -> Any:
        """Construct and cache the raw resource once on the target loop."""
        if self._raw is not None:
            return self._raw
        if self._raw_lock is None:
            self._raw_lock = asyncio.Lock()
        async with self._raw_lock:
            if self._raw is None:
                self._raw = self._raw_cls(self._config, **self._refs)
        return self._raw

    async def _async_execute_call(self, call: RuntimeCall) -> Any:
        """Execute a primitive call with async-first method selection."""
        raw = await self._get_raw()
        method = self._select_method(raw, call.method)

        result = method(*call.args, **call.kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    @staticmethod
    def _select_method(raw: Any, method_name: str) -> Any:
        """Prefer ``async_<method>`` for sync calls when the raw resource has it."""
        if not method_name.startswith("async_"):
            async_method = getattr(raw, f"async_{method_name}", None)
            if callable(async_method):
                return async_method
        return getattr(raw, method_name)

    async def _async_getattr_impl(self, name: str) -> Any:
        raw = await self._get_raw()
        return getattr(raw, name)

    async def _async_close_impl(self) -> None:
        """Close the raw resource if it has been constructed."""
        if self._raw is None:
            return
        async_close = getattr(self._raw, "async_close", None)
        if callable(async_close):
            result = async_close()
            if inspect.isawaitable(result):
                await result
            return
        close = getattr(self._raw, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        return
