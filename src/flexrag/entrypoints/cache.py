import json
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from flexrag.retriever.retriever_base import RETRIEVAL_CACHE
from flexrag.utils import Choices


@dataclass
class Config:
    export_path: str = MISSING
    action: Choices(["clear", "export", "_"]) = "_"  # type: ignore


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    match config.action:
        case "clear":
            RETRIEVAL_CACHE.clear()
        case "export":
            with open(config.export_path, "w", encoding="utf-8") as f:
                for data in RETRIEVAL_CACHE:
                    data["retrieved_contexts"] = RETRIEVAL_CACHE[data]
                    f.write(json.dumps(data) + "\n")
        case _:
            raise ValueError("No action specified")
    return


if __name__ == "__main__":
    main()
