from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from flexrag.retriever import FlexRetriever
from flexrag.retriever.index import MultiFieldIndexConfig, RetrieverIndexConfig


@dataclass
class Config(RetrieverIndexConfig, MultiFieldIndexConfig):
    index_name: str = MISSING
    retriever_path: str = MISSING
    rebuild: bool = False


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
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
