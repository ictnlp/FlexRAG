import logging
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))


from kylin.retriever import (
    BM25SRetriever,
    BM25SRetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
    ElasticRetriever,
    ElasticRetrieverConfig,
    MilvusRetriever,
    MilvusRetrieverConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from kylin.utils import Choices, read_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# fmt: off
@dataclass
class Config:
    retriever_type: Choices(["dense", "elastic", "milvus", "typesense", "bm25s"]) = "dense"  # type: ignore
    bm25s_config: BM25SRetrieverConfig = field(default_factory=BM25SRetrieverConfig)
    dense_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)
    milvus_config: MilvusRetrieverConfig = field(default_factory=MilvusRetrieverConfig)
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)
    corpus_path: list[str] = MISSING
    data_ranges: Optional[list[list[int]]] = field(default=None)
    full_text_field: str = "text"
    title_field: str = "title"
    section_field: str = "section"
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
        case "milvus":
            retriever = MilvusRetriever(cfg.milvus_config)
        case "typesense":
            retriever = TypesenseRetriever(cfg.typesense_config)

    # add passages
    if cfg.reinit and (len(retriever) > 0):
        logger.warning("Reinitializing retriever and removing all passages")
        retriever.clean()
    retriever.add_passages(
        passages=read_data(cfg.corpus_path, cfg.data_ranges),
        full_text_field=cfg.full_text_field,
        title_field=cfg.title_field,
        section_field=cfg.section_field,
    )
    retriever.close()
    return


if __name__ == "__main__":
    main()
