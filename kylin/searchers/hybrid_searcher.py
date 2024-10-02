from dataclasses import dataclass, field

from kylin.retriever import RetrievedContext
from kylin.utils import Choices

from .dense_searcher import DenseSearcher, DenseSearcherConfig
from .keyword_searcher import KeywordSearcher, KeywordSearcherConfig
from .searcher import (
    AgentSearcher,
    AgentSearcherConfig,
    BaseSearcher,
    Searchers,
    SearchHistory,
)
from .web_searcher import WebSearcher, WebSearcherConfig


@dataclass
class HybridSearcherConfig(AgentSearcherConfig):
    searchers: list[Choices(["keyword", "web", "dense"])] = field(default_factory=list)  # type: ignore
    keyword_searcher_config: KeywordSearcherConfig = field(default_factory=KeywordSearcherConfig)  # fmt: skip
    web_searcher_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    dense_searcher_config: DenseSearcherConfig = field(default_factory=DenseSearcherConfig)  # fmt: skip


@Searchers("hybrid", config_class=HybridSearcherConfig)
class HybridSearcher(AgentSearcher):
    is_hybrid = True

    def __init__(self, cfg: HybridSearcherConfig) -> None:
        super().__init__(cfg)
        # load searchers
        self.searchers = self.load_searchers(
            searchers=cfg.searchers,
            keyword_cfg=cfg.keyword_searcher_config,
            web_cfg=cfg.web_searcher_config,
            dense_cfg=cfg.dense_searcher_config,
        )
        return

    def load_searchers(
        self,
        searchers: list[str],
        keyword_cfg: KeywordSearcherConfig,
        web_cfg: WebSearcherConfig,
        dense_cfg: DenseSearcherConfig,
    ) -> dict[str, BaseSearcher]:
        searcher_list = {}
        for searcher in searchers:
            match searcher:
                case "keyword":
                    searcher_list[searcher] = KeywordSearcher(keyword_cfg)
                case "web":
                    searcher_list[searcher] = WebSearcher(web_cfg)
                case "dense":
                    searcher_list[searcher] = DenseSearcher(dense_cfg)
                case _:
                    raise ValueError(f"Searcher {searcher} not supported")
        return searcher_list

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        # search the question using sub-searchers
        contexts = []
        history = []
        for name, searcher in self.searchers.items():
            ctxs = searcher.search(question)[0]
            contexts.extend(ctxs)
            history.append(SearchHistory(query=question, contexts=ctxs))
        return contexts, history

    def close(self) -> None:
        for name in self.searchers:
            self.searchers[name].close()
        return
