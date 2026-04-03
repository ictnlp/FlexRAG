from flexrag.models.process_worker_pool import ProcessWorkerPoolClient


class ScorerWorkerPoolClient(ProcessWorkerPoolClient):
    """Scorer-specific worker-pool adapter built on ``ProcessWorkerPoolClient``."""

    async def score(self, pairs: list[tuple[str, str]]):
        return await self.call_available("score", pairs)
