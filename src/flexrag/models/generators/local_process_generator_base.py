from .async_generator_base import AsyncGeneratorBase
from .process_runtime import GeneratorWorkerPoolClient


class LocalProcessGeneratorBase(AsyncGeneratorBase):
    """Base class for local generators executed in worker subprocesses."""

    impl_cls: type | None = None

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return GeneratorWorkerPoolClient.from_config(self.impl_cls, config)

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return max(1, len(getattr(self._config, "device_id", [])) or 1)

    async def _async_generate_impl(
        self, client, prefixes: list[str], generation_config
    ):
        return await client.generate(prefixes, generation_config=generation_config)

    async def _async_chat_impl(self, client, messages, generation_config):
        return await client.chat(messages, generation_config=generation_config)
