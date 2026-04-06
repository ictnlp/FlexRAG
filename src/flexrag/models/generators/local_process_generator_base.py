from flexrag.models.process_worker_pool import ProcessWorkerPoolClient

from .generator_base import GeneratorBase


class LocalProcessGeneratorBase(GeneratorBase):
    """Base class for local generators executed in worker subprocesses."""

    impl_cls: type | None = None

    def _build_worker_device_groups(self, config) -> list[list[int] | None]:
        device_ids = list(getattr(config, "device_id", []))
        if not device_ids:
            return [None]
        return [[device_id] for device_id in device_ids]

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return ProcessWorkerPoolClient.from_worker_groups(
            self.impl_cls,
            config,
            self._build_worker_device_groups(config),
        )

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return max(1, len(self._build_worker_device_groups(self._config)))

    async def _async_generate_impl(
        self, client, prefixes: list[str], generation_config
    ):
        return await client.call_available(
            "generate",
            prefixes,
            generation_config=generation_config,
        )

    async def _async_chat_impl(self, client, messages, generation_config):
        return await client.call_available(
            "chat",
            messages,
            generation_config=generation_config,
        )
