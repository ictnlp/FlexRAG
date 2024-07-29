from dataclasses import dataclass, field

from kylin.retriever import RetrievedContext
from kylin.utils import Choices

from .bm25_searcher import BM25Searcher, BM25SearcherConfig
from .dense_searcher import DenseSearcher, DenseSearcherConfig
from .searcher import BaseSearcher, BaseSearcherConfig, Searchers
from .web_searcher import WebSearcher, WebSearcherConfig


@dataclass
class HybridSearcherConfig(BaseSearcherConfig):
    searchers: list[Choices(["bm25", "web", "dense"])] = field(default_factory=list)  # type: ignore
    bm25_searcher_config: BM25SearcherConfig = field(default_factory=BM25SearcherConfig)
    web_searcher_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    dense_searcher_config: DenseSearcherConfig = field(default_factory=DenseSearcherConfig)  # fmt: skip


@Searchers("hybrid", config_class=HybridSearcherConfig)
class HybridSearcher(BaseSearcher):
    def __init__(self, cfg: HybridSearcherConfig) -> None:
        super().__init__(cfg)
        # load searchers
        self.searchers = self.load_searchers(
            searchers=cfg.searchers,
            bm25_cfg=cfg.bm25_searcher_config,
            web_cfg=cfg.web_searcher_config,
            dense_cfg=cfg.dense_searcher_config,
        )
        return

    def load_searchers(
        self,
        searchers: list[str],
        bm25_cfg: BM25SearcherConfig,
        web_cfg: WebSearcherConfig,
        dense_cfg: DenseSearcherConfig,
    ) -> dict[str, BaseSearcher]:
        searcher_list = {}
        for searcher in searchers:
            match searcher:
                case "bm25":
                    searcher_list[searcher] = BM25Searcher(bm25_cfg)
                case "web":
                    searcher_list[searcher] = WebSearcher(web_cfg)
                case "dense":
                    searcher_list[searcher] = DenseSearcher(dense_cfg)
                case _:
                    raise ValueError(f"Searcher {searcher} not supported")
        return searcher_list

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        # search the question using sub-searchers
        contexts = []
        search_history = []
        for name, searcher in self.searchers.items():
            ctxs = searcher.search(question)[0]
            contexts.extend(ctxs)
            search_history.append(
                {
                    "searcher": name,
                    "context": ctxs,
                }
            )
        return contexts, search_history
