"""Generic worker-pool helpers built on top of ``ProcessWorkerClient``."""

import asyncio
from dataclasses import asdict

from .process_worker import ProcessWorkerClient, get_symbol_path


class ProcessWorkerPoolClient:
    """Dispatch request batches to a pool of identical worker subprocesses."""

    def __init__(
        self,
        impl_path: str,
        config_cls_path: str,
        config_data: dict,
        worker_visible_device_groups: list[list[int] | None],
    ) -> None:
        self._workers = [
            ProcessWorkerClient(
                impl_path,
                config_cls_path,
                config_data,
                visible_device_ids,
            )
            for visible_device_ids in worker_visible_device_groups
        ]
        self._available_workers: asyncio.Queue[ProcessWorkerClient] = asyncio.Queue()
        for worker in self._workers:
            self._available_workers.put_nowait(worker)
        return

    def __len__(self) -> int:
        """Return the number of worker subprocesses in the pool."""
        return len(self._workers)

    @classmethod
    def from_device_groups(
        cls,
        impl_cls: type,
        config,
        device_groups: list[list[int]] | None,
    ):
        """Create a worker pool from adapter-level device placement.

        ``None`` creates one worker that inherits the parent process environment.
        An empty list creates one CPU-only worker by setting
        ``CUDA_VISIBLE_DEVICES`` to an empty string. A non-empty list creates one
        worker per listed device group.
        """
        if device_groups is None:
            worker_visible_device_groups = [None]
        elif len(device_groups) == 0:
            worker_visible_device_groups = [[]]
        else:
            worker_visible_device_groups = [list(group) for group in device_groups]
        return cls.from_worker_groups(impl_cls, config, worker_visible_device_groups)

    @classmethod
    def from_worker_groups(
        cls,
        impl_cls: type,
        config,
        worker_visible_device_groups: list[list[int] | None],
    ):
        """Create a worker pool with explicit visible-device groups.

        :param impl_cls: Worker implementation class to instantiate in each
            subprocess.
        :param config: Dataclass configuration passed to each worker.
        :param worker_visible_device_groups: Per-worker visible device IDs.
            ``None`` creates a worker without device remapping.
        :return: A process worker pool client.
        """
        return cls(
            impl_path=get_symbol_path(impl_cls),
            config_cls_path=get_symbol_path(type(config)),
            config_data=asdict(config),
            worker_visible_device_groups=worker_visible_device_groups,
        )

    async def call_primary(self, attribute: str, *args, **kwargs):
        """Call an attribute on the primary worker.

        :param attribute: Method or attribute name to call through worker RPC.
        :param args: Positional arguments forwarded to the worker call.
        :param kwargs: Keyword arguments forwarded to the worker call.
        :return: Result returned by the primary worker.
        """
        return await self._workers[0].call(attribute, *args, **kwargs)

    async def call_available(self, attribute: str, *args, **kwargs):
        """Call an attribute on the next available worker.

        The selected worker is returned to the availability queue after the RPC
        finishes or raises.

        :param attribute: Method or attribute name to call through worker RPC.
        :param args: Positional arguments forwarded to the worker call.
        :param kwargs: Keyword arguments forwarded to the worker call.
        :return: Result returned by the selected worker.
        """
        if len(self._workers) == 1:
            return await self.call_primary(attribute, *args, **kwargs)

        worker = await self._available_workers.get()
        try:
            return await worker.call(attribute, *args, **kwargs)
        finally:
            self._available_workers.put_nowait(worker)

    async def close(self) -> None:
        """Close all worker subprocess clients.

        Worker close errors are intentionally suppressed so shutdown can
        continue across the whole pool.
        """
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
        return
