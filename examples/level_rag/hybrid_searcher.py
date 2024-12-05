from dataclasses import dataclass, field

from librarian.assistant import ASSISTANTS, SearchHistory
from librarian.retriever import RetrievedContext
from librarian.utils import Choices

from .dense_searcher import DenseSearcher, DenseSearcherConfig
from .keyword_searcher import KeywordSearcher, KeywordSearcherConfig
from .searcher import AgentSearcher, AgentSearcherConfig
from .web_searcher import WebSearcher, WebSearcherConfig


@dataclass
class HybridSearcherConfig(AgentSearcherConfig):
    searchers: list[Choices(["keyword", "web", "dense"])] = field(default_factory=list)  # type: ignore
    levelrag_keyword_config: KeywordSearcherConfig = field(default_factory=KeywordSearcherConfig)  # fmt: skip
    levelrag_web_config: WebSearcherConfig = field(default_factory=WebSearcherConfig)
    levelrag_dense_config: DenseSearcherConfig = field(default_factory=DenseSearcherConfig)  # fmt: skip


@ASSISTANTS("levelrag_hybrid", config_class=HybridSearcherConfig)
class HybridSearcher(AgentSearcher):
    def __init__(self, cfg: HybridSearcherConfig) -> None:
        super().__init__(cfg)
        # load searchers
        self.searchers = self.load_searchers(
            searchers=cfg.searchers,
            keyword_cfg=cfg.levelrag_keyword_config,
            web_cfg=cfg.levelrag_web_config,
            dense_cfg=cfg.levelrag_dense_config,
        )
        return

    def load_searchers(
        self,
        searchers: list[str],
        keyword_cfg: KeywordSearcherConfig,
        web_cfg: WebSearcherConfig,
        dense_cfg: DenseSearcherConfig,
    ) -> dict[str, AgentSearcher]:
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
            # ctxs = [self._change_field_name(ctx) for ctx in ctxs]
            contexts.extend(ctxs)
            history.append(SearchHistory(query=question, contexts=ctxs))
        return contexts, history

    def _change_field_name(self, context: RetrievedContext):
        for field in self.used_fields:
            if field not in context.data:
                context.data[field] = list(context.data.values())[0]
        return context
