from __future__ import annotations

import asyncio
import inspect
from typing import Any

from .base import RuntimeCall
from .scheduler import NoRetryPolicy, RateLimiter
from .target_base import RuntimeTargetBase


class DirectTarget(RuntimeTargetBase):
    """Runtime target that owns a raw resource in the current process.

    ``DirectTarget`` constructs the raw resource immediately and injects refs as
    real typed handles. Raw synchronous methods run in a worker thread from the
    target background loop; coroutine methods are awaited directly. The target
    remains the lifecycle owner and closes the raw resource when the manager
    closes the target.
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
    ) -> None:
        """Create a direct target around one in-process raw resource.

        :param raw_cls: Raw resource class to instantiate.
        :param config: Config object passed as the first constructor argument.
        :param refs: Constructor refs materialized as typed handles.
        :param batch_size: Public-call batch size exposed to handles.
        :param max_concurrency: Maximum primitive calls to run concurrently.
        :param rpm: Attempt-level request-per-minute limit. ``0`` disables
            rate limiting.
        """
        self._raw = raw_cls(config, **(refs or {}))
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            call_policy=NoRetryPolicy(RateLimiter(rpm=rpm)),
        )
        return

    async def _async_execute_call(self, call: RuntimeCall) -> Any:
        """Execute a primitive call against the in-process raw resource."""
        method = getattr(self._raw, call.method)
        if inspect.iscoroutinefunction(method):
            return await method(*call.args, **call.kwargs)
        result = await asyncio.to_thread(method, *call.args, **call.kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _async_getattr_impl(self, name: str) -> Any:
        return getattr(self._raw, name)

    async def _async_close_impl(self) -> None:
        """Close the raw resource using ``async_close`` or ``close`` if present."""
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
