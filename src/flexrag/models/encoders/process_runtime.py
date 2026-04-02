import asyncio
from dataclasses import asdict

import numpy as np

from flexrag.models.process_worker import ProcessWorkerClient, get_symbol_path


class EncoderWorkerPoolClient:
    """Encoder-specific worker-pool adapter built on ``ProcessWorkerClient``.

    This class owns a group of local worker subprocesses that all host the same
    encoder implementation. It adds encoder-specific operations on top of the
    generic single-worker RPC client:

    - ``call_primary`` forwards a request to the first worker, which is useful
      for metadata reads or methods that should not be faned out;
    - ``embedding_size`` caches the embedding dimension fetched from the primary
      worker;
    - ``encode`` treats its input as one request batch and dispatches that whole
      batch to a currently available worker.

    In other words, ``ProcessWorkerClient`` defines how to talk to one worker,
    while ``EncoderWorkerPoolClient`` defines how several identical workers are
    organized into one encoder runtime.
    """

    def __init__(
        self,
        impl_path: str,
        config_cls_path: str,
        config_data: dict,
        worker_device_ids: list[int | None],
    ) -> None:
        self._workers = [
            ProcessWorkerClient(impl_path, config_cls_path, config_data, gpu_id)
            for gpu_id in worker_device_ids
        ]
        self._available_workers: asyncio.Queue[ProcessWorkerClient] = asyncio.Queue()
        for worker in self._workers:
            self._available_workers.put_nowait(worker)
        self._embedding_size = None
        return

    @classmethod
    def from_config(cls, impl_cls: type, config):
        worker_device_ids = list(getattr(config, "device_id", [])) or [None]
        impl_path = get_symbol_path(impl_cls)
        config_cls = type(config)
        config_cls_path = get_symbol_path(config_cls)
        return cls(
            impl_path=impl_path,
            config_cls_path=config_cls_path,
            config_data=asdict(config),
            worker_device_ids=worker_device_ids,
        )

    async def call_primary(self, attribute: str, *args, **kwargs):
        return await self._workers[0].call(attribute, *args, **kwargs)

    async def embedding_size(self):
        if self._embedding_size is None:
            self._embedding_size = await self.call_primary("embedding_size")
        return self._embedding_size

    async def encode(self, texts: list[str]) -> np.ndarray:
        if len(self._workers) == 1:
            return await self.call_primary("encode", texts)

        worker = await self._available_workers.get()
        try:
            return await worker.call("encode", texts)
        finally:
            self._available_workers.put_nowait(worker)

    async def close(self) -> None:
        await asyncio.gather(
            *[asyncio.to_thread(worker.close) for worker in self._workers],
            return_exceptions=True,
        )
        return
