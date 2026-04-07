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

    @classmethod
    def from_config(cls, impl_cls: type, config):
        worker_visible_device_groups = [
            [gpu_id] for gpu_id in (list(getattr(config, "device_id", [])) or [])
        ] or [None]
        return cls.from_worker_groups(
            impl_cls,
            config,
            worker_visible_device_groups,
        )

    @classmethod
    def from_worker_groups(
        cls,
        impl_cls: type,
        config,
        worker_visible_device_groups: list[list[int] | None],
    ):
        return cls(
            impl_path=get_symbol_path(impl_cls),
            config_cls_path=get_symbol_path(type(config)),
            config_data=asdict(config),
            worker_visible_device_groups=worker_visible_device_groups,
        )

    async def call_primary(self, attribute: str, *args, **kwargs):
        return await self._workers[0].call(attribute, *args, **kwargs)

    async def call_available(self, attribute: str, *args, **kwargs):
        if len(self._workers) == 1:
            return await self.call_primary(attribute, *args, **kwargs)

        worker = await self._available_workers.get()
        try:
            return await worker.call(attribute, *args, **kwargs)
        finally:
            self._available_workers.put_nowait(worker)

    async def close(self) -> None:
        for worker in self._workers:
            try:
                worker.close()
            except Exception:
                pass
        return
