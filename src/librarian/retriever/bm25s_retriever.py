import os
from dataclasses import dataclass
from typing import Iterable, Optional

import bm25s
from omegaconf import MISSING

from librarian.utils import Choices, LOGGER_MANAGER, TIME_METER

from .retriever_base import (
    RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = LOGGER_MANAGER.get_logger("librarian.retrievers.bm25s")


@dataclass
class BM25SRetrieverConfig(LocalRetrieverConfig):
    database_path: str = MISSING
    method: Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"]) = "lucene"  # type: ignore
    idf_method: Optional[Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"])] = None  # type: ignore
    backend: Choices(["numpy", "numba", "auto"]) = "auto"  # type: ignore
    k1: float = 1.5
    b: float = 0.75
    delta: float = 0.5
    lang: str = "english"
    indexed_fields: Optional[list[str]] = None


@RETRIEVERS("bm25s", config_class=BM25SRetrieverConfig)
class BM25SRetriever(LocalRetriever):
    name = "BM25SSearch"

    def __init__(self, cfg: BM25SRetrieverConfig) -> None:
        super().__init__(cfg)
        # set basic args
        try:
            import Stemmer

            self._stemmer = Stemmer.Stemmer(cfg.lang)
        except:
            self._stemmer = None

        # load retriever
        self.database_path = cfg.database_path
        if os.path.exists(self.database_path) and bool(os.listdir(self.database_path)):
            self._retriever = bm25s.BM25.load(
                self.database_path,
                mmap=True,
                load_corpus=True,
            )
        else:
            os.makedirs(self.database_path, exist_ok=True)
            self._retriever = bm25s.BM25(
                method=cfg.method,
                idf_method=cfg.idf_method,
                backend=cfg.backend,
                k1=cfg.k1,
                b=cfg.b,
                delta=cfg.delta,
            )
        self._lang = cfg.lang
        self._indexed_fields = cfg.indexed_fields
        return

    @TIME_METER("bm25s_retriever", "add-passages")
    def add_passages(self, passages: Iterable[dict[str, str]]):
        logger.warning(
            "bm25s Retriever does not support add passages. This function will build the index from scratch."
        )
        passages = list(passages)
        if len(self._indexed_fields) == 1:
            indexed = [p[self._indexed_fields[0]] for p in passages]
        else:
            indexed = [" ".join([p[f] for f in self._indexed_fields]) for p in passages]
        indexed_tokens = bm25s.tokenize(
            indexed, stopwords=self._lang, stemmer=self._stemmer
        )
        self._retriever.index(indexed_tokens)
        self._retriever.corpus = passages
        self._retriever.save(self.database_path, corpus=passages)
        return

    @TIME_METER("bm25s_retriever", "search")
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # retrieve
        query_tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        contexts, scores = self._retriever.retrieve(
            query_tokens,
            k=search_kwargs.get("top_k", self.top_k),
            show_progress=False,
            **search_kwargs,
        )

        # form final results
        results = []
        for q, ctxs, score in zip(query, contexts, scores):
            results.append(
                [
                    RetrievedContext(
                        retriever=self.name,
                        query=q,
                        data=ctx,
                        score=score[i],
                    )
                    for i, ctx in enumerate(ctxs)
                ]
            )
        return results

    def clean(self) -> None:
        del self._retriever.scores
        del self._retriever.vocab_dict
        return

    def __len__(self) -> int:
        if hasattr(self._retriever, "scores"):
            return self._retriever.scores.get("num_docs", 0)
        return 0

    @property
    def fields(self) -> list[str]:
        if self._retriever.corpus is not None:
            return self._retriever.corpus[0].keys()
        return []
