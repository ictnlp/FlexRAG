from __future__ import annotations

import asyncio
import itertools
from typing import Any

from ..errors import raise_runtime_error
from ..refs import ResourceRefDescriptor
from .base import RuntimeCall


class ParentProxyTarget:
    """Worker-side target that proxies handle calls back to the parent manager.

    A process worker wraps dependency handles around this target. Calls are sent
    to the parent process, where the owning ``ResourceManager`` executes them on
    the real dependency target. The proxy itself owns no resource lifecycle.
    """

    _ids = itertools.count(1)

    def __init__(
        self,
        descriptor: ResourceRefDescriptor,
        conn: Any,
    ) -> None:
        self.descriptor = descriptor
        self.conn = conn
        return

    @property
    def batch_size(self) -> int:
        """Return the referenced dependency batch size."""
        return self.descriptor.batch_size

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously proxy one dependency method call to the parent."""
        request_id = f"dep-{next(self._ids)}"
        self.conn.send(
            {
                "kind": "dependency_call",
                "id": request_id,
                "resource": self.descriptor.name,
                "method": method,
                "args": args,
                "kwargs": kwargs,
            }
        )
        response = self.conn.recv()
        if response.get("kind") != "dependency_response" or response.get(
            "id"
        ) != request_id:
            raise RuntimeError(f"Unexpected dependency response: {response!r}")
        if not response.get("ok", False):
            raise_runtime_error(response)
        return response.get("data")

    async def async_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously proxy one dependency method call to the parent."""
        return await asyncio.to_thread(self.call, method, *args, **kwargs)

    def batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Synchronously proxy dependency batch calls to the parent."""
        request_id = f"dep-batch-{next(self._ids)}"
        self.conn.send(
            {
                "kind": "dependency_batch_call",
                "id": request_id,
                "resource": self.descriptor.name,
                "calls": calls,
                "log_interval": log_interval,
                "display": display,
                "desc": desc,
            }
        )
        response = self.conn.recv()
        if response.get("kind") != "dependency_response" or response.get(
            "id"
        ) != request_id:
            raise RuntimeError(f"Unexpected dependency response: {response!r}")
        if not response.get("ok", False):
            raise_runtime_error(response)
        return response.get("data")

    async def async_batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Asynchronously proxy dependency batch calls to the parent."""
        return await asyncio.to_thread(
            self.batch_call,
            calls,
            log_interval=log_interval,
            display=display,
            desc=desc,
        )

    def getattr(self, name: str) -> Any:
        """Synchronously proxy dependency attribute access to the parent."""
        return self.call("__getattr__", name)

    async def async_getattr(self, name: str) -> Any:
        """Asynchronously proxy dependency attribute access to the parent."""
        return await self.async_call("__getattr__", name)

    def close(self) -> None:
        """No-op close; the parent manager owns the referenced resource."""
        return

    async def async_close(self) -> None:
        """No-op async close; the parent manager owns the referenced resource."""
        return
