from flexrag.models.process_worker_pool import ProcessWorkerPoolClient

from .scorer_base import PairScorerBase


class LocalProcessScorerBase(PairScorerBase):
    """Base class for local scorers executed in worker subprocesses."""

    impl_cls: type | None = None

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return ProcessWorkerPoolClient.from_config(self.impl_cls, config)

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return max(1, len(getattr(self._config, "device_id", [])) or 1)

    async def _async_score_impl(self, client, pairs: list[tuple[str, str]]):
        return await client.call_available("score", pairs)
