from dataclasses import dataclass, field

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from flexrag.datasets import RAGCorpusDataset, RAGCorpusDatasetConfig
from flexrag.retriever import (
    BM25SRetriever,
    BM25SRetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
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
    retriever_type: Choices(["dense", "elastic", "typesense", "bm25s"]) = "dense"  # type: ignore
    bm25s_config: BM25SRetrieverConfig = field(default_factory=BM25SRetrieverConfig)
    dense_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)
    reinit: bool = False
# fmt: on


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    default_cfg = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(default_cfg, cfg)

    # load retriever
    match cfg.retriever_type:
        case "bm25s":
            retriever = BM25SRetriever(cfg.bm25s_config)
        case "dense":
            retriever = DenseRetriever(cfg.dense_config)
        case "elastic":
            retriever = ElasticRetriever(cfg.elastic_config)
        case "typesense":
            retriever = TypesenseRetriever(cfg.typesense_config)
        case _:
            raise ValueError(f"Unsupported retriever type: {cfg.retriever_type}")

    # add passages
    if cfg.reinit and (len(retriever) > 0):
        logger.warning("Reinitializing retriever and removing all passages")
        retriever.clean()

    retriever.add_passages(passages=RAGCorpusDataset(cfg))
    return


if __name__ == "__main__":
    main()
