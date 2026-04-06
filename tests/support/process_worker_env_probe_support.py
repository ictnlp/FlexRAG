import os
from dataclasses import dataclass, field

IMPORT_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")


@dataclass
class ProcessWorkerEnvProbeConfig:
    device_id: list[int] = field(default_factory=list)


class ProcessWorkerEnvProbeImpl:
    def __init__(self, config: ProcessWorkerEnvProbeConfig) -> None:
        self.config = config
        return

    def report(self) -> dict:
        return {
            "import_visible_devices": IMPORT_VISIBLE_DEVICES,
            "runtime_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "config_device_id": list(self.config.device_id),
        }
