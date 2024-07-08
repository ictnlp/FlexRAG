import logging
from dataclasses import dataclass, field
from typing import Optional

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from kylin.assistant import Assistant, AssistantConfig
from kylin.searchers import (
    BM25Searcher,
    BM25SearcherConfig,
    WebSearcher,
    WebSearcherConfig,
    DenseSearcher,
    DenseSearcherConfig,
)
from kylin.utils import Choices

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    assistant_config: AssistantConfig = field(default_factory=AssistantConfig)
    searcher_type: Optional[Choices(["bm25", "web", "dense"])] = None  # type: ignore
    bm25_searcher_config: BM25SearcherConfig = field(default_factory=BM25SearcherConfig)
    web_searcher_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    dense_searcher_config: DenseSearcherConfig = field(default_factory=DenseSearcherConfig)  # fmt: skip
    response_type: Choices(["short", "long"]) = "short"  # type: ignore


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.1", config_path=None, config_name="default")
def main(config: Config):
    # merge config
    default_cfg = OmegaConf.structured(Config)
    config = OmegaConf.merge(default_cfg, config)
    logging.info(f"Configs:\n{OmegaConf.to_yaml(config)}")

    # load searcher
    match config.searcher_type:
        case "bm25":
            searcher = BM25Searcher(config.bm25_searcher_config)
        case "web":
            searcher = WebSearcher(config.web_searcher_config)
        case "dense":
            searcher = DenseSearcher(config.dense_searcher_config)
        case None:
            searcher = None
        case _:
            raise ValueError(f"Invalid searcher type: {config.searcher_type}")

    # load assistant
    assistant = Assistant(config.assistant_config)

    while True:
        query = input("Ask me anything(type `quit` to quit): ")
        if query == "quit":
            break
        # search
        if searcher is not None:
            ctxs, _ = searcher.search(query)
        else:
            ctxs = []
        # generate
        r, _ = assistant.answer(question=query, contexts=ctxs)
        print(f"Response: {r}")

    if searcher is not None:
        searcher.close()
    return


if __name__ == "__main__":
    main()
