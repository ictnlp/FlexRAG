from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from .base import RuntimeCall
from .event_loop import BackgroundEventLoop
from .scheduler import CallPolicy, RuntimeBatchScheduler


class RuntimeTargetBase(ABC):
    """Shared scheduling, bridge, and lifecycle implementation for targets.

    Subclasses only implement raw call execution, attribute reads, and raw
    cleanup. This base class owns the background event loop, batch scheduler,
    call policy, synchronous wrappers, and closed-state checks.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        max_concurrency: int,
        call_policy: CallPolicy,
    ) -> None:
        """Initialize the common target runtime shell.

        :param batch_size: Public-call batch size exposed through handles.
        :param max_concurrency: Maximum primitive calls the scheduler may run
            concurrently.
        :param call_policy: Attempt-level policy used for each primitive call.
        """
        self._batch_size = batch_size
        self._loop = BackgroundEventLoop()
        self._scheduler = RuntimeBatchScheduler(max_concurrency=max_concurrency)
        self._call_policy = call_policy
        self._closed = False
        return

    @property
    def batch_size(self) -> int:
        """Return the handle-visible batch size for this target."""
        return self._batch_size

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed.")
        return

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously execute a single primitive method call.

        :param method: Raw resource method name.
        :param args: Positional arguments for the raw method.
        :param kwargs: Keyword arguments for the raw method.
        :returns: Raw method result.
        :raises RuntimeError: If the target has been closed.
        """
        return self.batch_call([RuntimeCall(method, args, kwargs)])[0]

    async def async_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously execute a single primitive method call.

        :param method: Raw resource method name.
        :param args: Positional arguments for the raw method.
        :param kwargs: Keyword arguments for the raw method.
        :returns: Raw method result.
        :raises RuntimeError: If the target has been closed.
        """
        return (await self.async_batch_call([RuntimeCall(method, args, kwargs)]))[0]

    def batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Synchronously execute primitive calls through the runtime scheduler.

        :param calls: Primitive calls to execute.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :param desc: Progress description used after each primitive call.
        :returns: Results in the same order as ``calls``.
        :raises RuntimeError: If the target has been closed.
        """
        self._ensure_open()
        return self._loop.run_async(
            self._async_batch_call_impl(
                calls,
                log_interval=log_interval,
                display=display,
                desc=desc,
            )
        ).result()

    async def async_batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Asynchronously execute primitive calls through the runtime scheduler.

        :param calls: Primitive calls to execute.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :param desc: Progress description used after each primitive call.
        :returns: Results in the same order as ``calls``.
        :raises RuntimeError: If the target has been closed.
        """
        self._ensure_open()
        future = self._loop.run_async(
            self._async_batch_call_impl(
                calls,
                log_interval=log_interval,
                display=display,
                desc=desc,
            )
        )
        return await asyncio.wrap_future(future)

    async def _async_batch_call_impl(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int,
        display: str,
        desc: str,
    ) -> list[Any]:
        """Run calls on the background loop with this target's call policy."""
        return await self._scheduler.run(
            calls,
            self._async_execute_call,
            self._call_policy,
            log_interval=log_interval,
            display=display,
            desc=desc,
        )

    def getattr(self, name: str) -> Any:
        """Synchronously read an attribute from the raw resource.

        :param name: Attribute name.
        :returns: Attribute value.
        :raises RuntimeError: If the target has been closed.
        """
        self._ensure_open()
        return self._loop.run_async(self._async_getattr_impl(name)).result()

    async def async_getattr(self, name: str) -> Any:
        """Asynchronously read an attribute from the raw resource.

        :param name: Attribute name.
        :returns: Attribute value.
        :raises RuntimeError: If the target has been closed.
        """
        self._ensure_open()
        future = self._loop.run_async(self._async_getattr_impl(name))
        return await asyncio.wrap_future(future)

    def close(self) -> None:
        """Synchronously close the raw resource and stop the background loop."""
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.run_async(self._async_close_impl()).result()
        finally:
            self._loop.stop()
        return

    async def async_close(self) -> None:
        """Asynchronously close the raw resource and stop the background loop."""
        if self._closed:
            return
        self._closed = True
        try:
            future = self._loop.run_async(self._async_close_impl())
            await asyncio.wrap_future(future)
        finally:
            self._loop.stop()
        return

    @abstractmethod
    async def _async_execute_call(self, call: RuntimeCall) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def _async_getattr_impl(self, name: str) -> Any:
        raise NotImplementedError

    @abstractmethod
    async def _async_close_impl(self) -> None:
        raise NotImplementedError
