import asyncio
import json

from flexrag.resources.runtime.process_worker_pool import ProcessWorkerPoolClient
from tests.support.process.process_worker_env_probe_support import (
    ProcessWorkerEnvProbeConfig,
    ProcessWorkerEnvProbeImpl,
)


async def main() -> None:
    async def collect(device_groups, calls: int):
        pool = ProcessWorkerPoolClient.from_device_groups(
            ProcessWorkerEnvProbeImpl,
            ProcessWorkerEnvProbeConfig(),
            device_groups,
        )
        try:
            return [await pool.call_available("report") for _ in range(calls)]
        finally:
            await pool.close()

    reports = {
        "explicit": await collect([[0], [1], [2], [3]], 4),
        "cpu": await collect([], 1),
        "inherit": await collect(None, 1),
    }
    print(json.dumps(reports, ensure_ascii=False))
    return


if __name__ == "__main__":
    asyncio.run(main())
