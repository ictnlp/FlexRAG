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
        self,
        question: str,
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        # retrieve
        history: list[SearchHistory] = []
        ctxs = self.retriever.search(
            query=[question],
            top_k=self.retriever_top_k,
            disable_cache=self.disable_cache,
        )[0]
        history.append(SearchHistory(query=question, contexts=ctxs))

        # rerank
        if self.reranker is not None:
            results = self.reranker.rank(question, ctxs)
            ctxs = results.candidates
            history.append(SearchHistory(query=question, contexts=ctxs))
        return ctxs, history

    def search_batch(
        self,
        questions: list[str],
    ) -> tuple[list[list[RetrievedContext]], list[list[SearchHistory]]]:
        # retrieve
        history: list[list[SearchHistory]] = []
        ctxs = self.retriever.search(
            query=questions,
            top_k=self.retriever_top_k,
            disable_cache=self.disable_cache,
        )
        history.append(
            [SearchHistory(query=q, contexts=ctx) for q, ctx in zip(questions, ctxs)]
        )

        # rerank
        new_ctxs = []
        if self.reranker is not None:
            history.append([])
            for q, ctx in zip(questions, ctxs):
                result = self.reranker.rank(q, ctx)
                ctx = result.candidates
                new_ctxs.append(ctx)
                history[-1].append(SearchHistory(query=q, contexts=ctx))
        return ctxs, history

    def close(self) -> None:
        self.retriever.close()
        return
