import os
from dataclasses import dataclass

IMPORT_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


@dataclass
class ProcessWorkerEnvProbeConfig:
    pass


class ProcessWorkerEnvProbeImpl:
    def __init__(self, config: ProcessWorkerEnvProbeConfig) -> None:
        self.config = config
        return

    def report(self) -> dict:
        return {
            "import_visible_devices": IMPORT_VISIBLE_DEVICES,
            "runtime_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
