from flexrag.models.process_worker_pool import ProcessWorkerPoolClient


class GeneratorWorkerPoolClient(ProcessWorkerPoolClient):
    """Generator-specific worker-pool adapter built on ``ProcessWorkerPoolClient``."""

    async def generate(self, prefixes: list[str], generation_config=None):
        return await self.call_available(
            "generate",
            prefixes,
            generation_config=generation_config,
        )

    async def chat(self, messages, generation_config=None):
        return await self.call_available(
            "chat",
            messages,
            generation_config=generation_config,
        )
