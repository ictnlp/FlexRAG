from .async_encoder_base import AsyncEncoderBase
from .process_runtime import EncoderWorkerPoolClient


class LocalProcessEncoderBase(AsyncEncoderBase):
    """Base class for local encoders executed in worker subprocesses.

    ``LocalProcessEncoderBase`` turns a concrete single-process encoder
    implementation into a proxy object that exposes the normal encoder API in
    the parent process while delegating real work to a local worker pool.

    Subclasses are expected to set ``impl_cls`` to the concrete encoder
    implementation class. During client creation, the base class builds an
    :class:`EncoderWorkerPoolClient` from that implementation and the current
    config. The worker pool is responsible for starting subprocesses, binding
    devices, and executing ``encode`` requests out of process.

    The public ``encode`` / ``async_encode`` methods are inherited from
    ``AsyncEncoderBase`` and ultimately call ``client.encode(...)`` for one
    batch of texts. For operations that should always run on a single worker,
    such as reading metadata or invoking model-specific helper methods,
    ``_async_call_primary`` and ``_call_primary`` bypass the pool dispatcher and
    forward the request directly to the primary worker.
    """

    impl_cls: type | None = None

    def __init__(self, config):
        super().__init__(config)
        self._embedding_size = None
        return

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        client = EncoderWorkerPoolClient.from_config(self.impl_cls, config)
        self._embedding_size = await client.embedding_size()
        return client

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return max(1, len(getattr(self._config, "device_id", [])) or 1)

    async def _async_encode_impl(self, client, texts: list[str]):
        return await client.encode(texts)

    async def _async_call_primary(self, attribute: str, *args, **kwargs):
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            return await client.call_primary(attribute, *args, **kwargs)

    def _call_primary(self, attribute: str, *args, **kwargs):
        return self._run_coroutine_sync(
            self._async_call_primary(attribute, *args, **kwargs)
        )

    @property
    def embedding_size(self):
        if self._embedding_size is None:
            self._embedding_size = self._call_primary("embedding_size")
        return self._embedding_size
