from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.retriever import (
    FlexRetriever,
    FlexRetrieverConfig,
    ElasticRetriever,
    ElasticRetrieverConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from flexrag.utils import LOGGER_MANAGER, Choices

logger = LOGGER_MANAGER.get_logger("flexrag.prepare_index")


# fmt: off
@dataclass
class Config(RAGCorpusDatasetConfig):
    # retriever configs
    retriever_type: Choices(["flex", "elastic", "typesense"]) = "flex"  # type: ignore
    flex_config: FlexRetrieverConfig = field(default_factory=FlexRetrieverConfig)
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)
    reinit: bool = False
# fmt: on


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    # load retriever
    match cfg.retriever_type:
        case "flex":
            retriever = FlexRetriever(cfg.flex_config)
        case "elastic":
            retriever = ElasticRetriever(cfg.elastic_config)
        case "typesense":
            retriever = TypesenseRetriever(cfg.typesense_config)
        case _:
            raise ValueError(f"Unsupported retriever type: {cfg.retriever_type}")

    # add passages
    if cfg.reinit and (len(retriever) > 0):
        logger.warning("Reinitializing retriever and removing all passages")
        retriever.clear()

    retriever.add_passages(passages=RAGCorpusDataset(cfg))
    return


if __name__ == "__main__":
    main()
