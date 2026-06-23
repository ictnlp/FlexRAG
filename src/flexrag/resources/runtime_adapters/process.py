from typing import Any

from flexrag.runtime.async_client import AsyncClientMixin, ConfigT
from flexrag.runtime.process_worker_pool import ProcessWorkerPoolClient


class ProcessRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Process-backed runtime for local raw resources.

    The runtime owns worker-pool creation, device-group placement, worker RPC,
    and lifecycle. Interface-specific input handling, batching, progress, and
    result merging belong to invocation objects.
    """

    impl_cls: type[Any] | None = None

    def __init__(
        self,
        config: ConfigT,
        impl_cls: type[Any] | None = None,
        *,
        device_groups: list[list[int]] | None = None,
    ) -> None:
        """Create a process runtime.

        :param config: Configuration passed to worker raw implementations.
        :param impl_cls: Optional raw implementation class. When omitted,
            subclasses must set ``impl_cls``.
        :param device_groups: Worker device placement. ``None`` creates one
            worker inheriting the current environment, ``[]`` creates one
            CPU-only worker, and non-empty groups create one worker per group.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        self._device_groups = device_groups
        self._worker_count = 1
        return

    async def _create_client(self, config: ConfigT) -> ProcessWorkerPoolClient:
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        client = ProcessWorkerPoolClient.from_device_groups(
            self.impl_cls,
            config,
            self._device_groups,
        )
        self._worker_count = len(client)
        return client

    async def _close_client(self, client: ProcessWorkerPoolClient) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return self._worker_count

    def run_sync(self, coro):
        """Run a coroutine on the managed runtime loop synchronously."""
        return self._run_coroutine_sync(coro)

    async def run_async(self, coro):
        """Run a coroutine on the managed runtime loop asynchronously."""
        return await self._run_coroutine_async(coro)

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the next available worker.

        :param method: Worker method or attribute name.
        :param args: Positional arguments forwarded to the worker RPC.
        :param kwargs: Keyword arguments forwarded to the worker RPC.
        :return: Worker RPC return value.
        """
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            return await client.call_available(method, *args, **kwargs)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously call a method on the next available worker."""
        return self.run_sync(self.acall(method, *args, **kwargs))

    async def acall_primary(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method or attribute on the primary worker."""
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            return await client.call_primary(method, *args, **kwargs)

    def call_primary(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously call a method or attribute on the primary worker."""
        return self.run_sync(self.acall_primary(method, *args, **kwargs))

    async def agetattr(self, name: str) -> Any:
        """Return an attribute from the primary worker."""
        return await self.acall_primary(name)

    def getattr(self, name: str) -> Any:
        """Synchronously return an attribute from the primary worker."""
        return self.call_primary(name)
