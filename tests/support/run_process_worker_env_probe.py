import asyncio
import json

from flexrag.runtime.process_worker_pool import ProcessWorkerPoolClient
from tests.support.process_worker_env_probe_support import (
    ProcessWorkerEnvProbeConfig,
    ProcessWorkerEnvProbeImpl,
)


async def main() -> None:
    config = ProcessWorkerEnvProbeConfig(device_id=[0, 1, 2, 3])
    pool = ProcessWorkerPoolClient.from_worker_groups(
        ProcessWorkerEnvProbeImpl,
        config,
        [[0], [1], [2], [3]],
    )
    try:
        reports = [await pool.call_available("report") for _ in range(4)]
    finally:
        await pool.close()
    print(json.dumps(reports, ensure_ascii=False))
    return


if __name__ == "__main__":
    asyncio.run(main())
