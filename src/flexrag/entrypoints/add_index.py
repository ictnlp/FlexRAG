from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.retriever import FlexRetriever
from flexrag.retriever.index import MultiFieldIndexConfig, RetrieverIndexConfig
from flexrag.utils import configure, extract_config


@configure
class Config(RetrieverIndexConfig, MultiFieldIndexConfig):
    index_name: Optional[str] = None
    retriever_path: Optional[str] = None
    rebuild: bool = False


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    cfg = extract_config(cfg, Config)
    assert cfg.index_name is not None, "index_name must be provided"
    assert cfg.retriever_path is not None, "retriever_path must be provided"
    retriever: FlexRetriever = FlexRetriever.load_from_local(cfg.retriever_path)

    # remove index
    if cfg.rebuild:
        retriever.remove_index(cfg.index_name)

    # add index
    retriever.add_index(
        index_name=cfg.index_name,
        index_config=cfg,
        indexed_fields_config=cfg,
    )
    return


if __name__ == "__main__":
    main()
