import json
from typing import Annotated, Optional

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.common import Choices, configure, extract_config
from flexrag.common.runtime_cache import get_runtime_cache

RETRIEVAL_CACHE_NAMESPACE = "retrieval.search"


@configure
class Config:
    export_path: Optional[str] = None
    action: Annotated[str, Choices("clear", "export", "_")] = "_"


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(config: Config):
    config = extract_config(config, Config)
    cache = get_runtime_cache(RETRIEVAL_CACHE_NAMESPACE)
    match config.action:
        case "clear":
            cache.clear()
        case "export":
            if config.export_path is None:
                raise ValueError("`export_path` must be provided for export.")
            with open(config.export_path, "w", encoding="utf-8") as f:
                for item in cache.items():
                    data = {
                        "key": item["key"],
                        "retrieved_contexts": item["value"],
                        "metadata": item["metadata"],
                        "created_at": item["created_at"],
                        "accessed_at": item["accessed_at"],
                        "size_bytes": item["size_bytes"],
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + "\n")
        case _:
            raise ValueError("No action specified")
    return


if __name__ == "__main__":
    main()
