from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from librarian.data import (
    LineDelimitedDataset,
    TextProcessPipeline,
    TextProcessPipelineConfig,
)
from librarian.retriever import (
    BM25SRetriever,
    BM25SRetrieverConfig,
    DenseRetriever,
    DenseRetrieverConfig,
    ElasticRetriever,
    ElasticRetrieverConfig,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from librarian.utils import LOGGER_MANAGER, Choices

logger = LOGGER_MANAGER.get_logger("librarian.prepare_index")


# fmt: off
@dataclass
class Config:
    # retriever configs
    retriever_type: Choices(["dense", "elastic", "milvus", "typesense", "bm25s", "lancedb"]) = "dense"  # type: ignore
    bm25s_config: BM25SRetrieverConfig = field(default_factory=BM25SRetrieverConfig)
    dense_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)
    reinit: bool = False
    # corpus configs
    corpus_path: list[str] = MISSING
    data_ranges: Optional[list[list[int]]] = field(default=None)
    saving_fields: list[str] = field(default_factory=list)
    # corpus process configs
    text_process_pipeline: TextProcessPipelineConfig = field(default_factory=TextProcessPipelineConfig)  # type: ignore
    text_process_fields: list[str] = field(default_factory=list)
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

    # prepare data iterator
    text_processor = TextProcessPipeline(cfg.text_process_pipeline)

    def prepare_data():
        for item in LineDelimitedDataset(cfg.corpus_path, cfg.data_ranges):
            # remove unused fields
            item = {key: item.get(key, "") for key in cfg.saving_fields}
            # preprocess text fields
            for key in cfg.text_process_fields:
                text = text_processor(item[key])
                if text is None:
                    text = ""
                item[key] = text
            yield item

    retriever.add_passages(passages=prepare_data())
    return


if __name__ == "__main__":
    main()
