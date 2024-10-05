import logging
from dataclasses import dataclass
from typing import Optional

from kylin.ranker import RankerConfig, load_ranker, Rankers
from kylin.retriever import RetrieverConfig, load_retriever, RetrievedContext
from kylin.utils import Choices

from .searcher import BaseSearcher, BaseSearcherConfig, Searchers, SearchHistory


logger = logging.getLogger("NaiveSearcher")


@dataclass
class NaiveSearcherConfig(BaseSearcherConfig, RetrieverConfig, RankerConfig):
    ranker_type: Optional[Choices(Rankers.names)] = None  # type: ignore
    ...


@Searchers("naive", config_class=NaiveSearcherConfig)
class NaiveSearcher(BaseSearcher):
    is_hybrid = False

    def __init__(self, cfg: NaiveSearcherConfig) -> None:
        super().__init__(cfg)
        # load retriever
        self.retriever = load_retriever(cfg)
        if cfg.ranker_type is not None:
            self.reranker = load_ranker(cfg)
        else:
            self.reranker = None
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

        if self.reranker is not None:
            results = self.reranker.rank(question, ctxs)
            ctxs = results.candidates
            history.append(SearchHistory(query=question, contexts=ctxs))
        return ctxs, history

    def close(self) -> None:
        self.retriever.close()
        return
