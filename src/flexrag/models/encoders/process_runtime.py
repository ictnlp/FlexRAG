from flexrag.models.process_worker_pool import ProcessWorkerPoolClient


class EncoderWorkerPoolClient(ProcessWorkerPoolClient):
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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._embedding_size = None
        return

    async def embedding_size(self):
        if self._embedding_size is None:
            self._embedding_size = await self.call_primary("embedding_size")
        return self._embedding_size

    async def encode(self, texts: list[str]):
        return await self.call_available("encode", texts)
