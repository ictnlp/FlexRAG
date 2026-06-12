from dataclasses import field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.common import configure, extract_config
from flexrag.models import ENCODERS, EncoderConfig
from flexrag.retrievers import FlexRetriever
from flexrag.retrievers.index import (
    RETRIEVER_INDEX,
    MultiFieldIndex,
    MultiFieldIndexConfig,
    RetrieverIndexConfig,
)
from flexrag.retrievers.index.index_base import DenseIndexBase


@configure
class Config(RetrieverIndexConfig, MultiFieldIndexConfig):
    index_name: Optional[str] = None
    retriever_path: Optional[str] = None
    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    passage_encoder_config: Optional[EncoderConfig] = None
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
    index_kwargs = {}
    index_cls = RETRIEVER_INDEX[str(cfg.index_type)]["item"]
    if issubclass(index_cls, DenseIndexBase):
        query_encoder = ENCODERS.load(cfg.query_encoder_config)
        if query_encoder is None:
            raise ValueError("query_encoder_config must be configured for dense index.")
        index_kwargs["query_encoder"] = query_encoder
        if cfg.passage_encoder_config is not None:
            passage_encoder = ENCODERS.load(cfg.passage_encoder_config)
            if passage_encoder is not None:
                index_kwargs["passage_encoder"] = passage_encoder
    base_index = RETRIEVER_INDEX.load(cfg, **index_kwargs)
    retriever.add_index(cfg.index_name, MultiFieldIndex(cfg, base_index))
    return


if __name__ == "__main__":
    main()
