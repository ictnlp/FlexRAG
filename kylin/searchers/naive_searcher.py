import logging
from dataclasses import dataclass

from kylin.retriever import RetrieverConfig, load_retriever, RetrievedContext

from .searcher import BaseSearcher, BaseSearcherConfig, Searchers, SearchHistory


logger = logging.getLogger("NaiveSearcher")


@dataclass
class NaiveSearcherConfig(BaseSearcherConfig, RetrieverConfig): ...


@Searchers("naive", config_class=NaiveSearcherConfig)
class NaiveSearcher(BaseSearcher):
    is_hybrid = False

    def __init__(self, cfg: NaiveSearcherConfig) -> None:
        super().__init__(cfg)
        # load retriever
        self.retriever = load_retriever(cfg)
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        # begin adaptive search
        ctxs = []
        history: list[SearchHistory] = []
        ctxs = self.retriever.search(
            query=[question],
            top_k=self.retriever_top_k,
            disable_cache=self.disable_cache,
        )[0]
        history.append(SearchHistory(query=question, contexts=ctxs))
        return ctxs, history

    def close(self) -> None:
        self.retriever.close()
        return
