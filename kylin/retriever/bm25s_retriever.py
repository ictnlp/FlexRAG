import logging
import os
from dataclasses import dataclass
from typing import Iterable, Optional

import bm25s
from omegaconf import MISSING

from kylin.utils import Choices

from .fingerprint import Fingerprint
from .retriever_base import (
    SPARSE_RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = logging.getLogger("BM25SRetriever")


@dataclass
class BM25SRetrieverConfig(LocalRetrieverConfig):
    database_path: str = MISSING
    method: Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"]) = "lucene"  # type: ignore
    idf_method: Optional[Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"])] = None  # type: ignore
    delta: float = 0.5
    lang: str = "english"
    index_name: str = "documents"


@SPARSE_RETRIEVERS("bm25s", config_class=BM25SRetrieverConfig)
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
        if os.path.exists(cfg.database_path):
            self._retriever = bm25s.BM25.load(
                cfg.database_path, mmap=True, load_corpus=True
            )
        else:
            os.makedirs(cfg.database_path)
            self._retriever = bm25s.BM25(
                method=cfg.method,
                idf_method=cfg.idf_method,
                delta=cfg.delta,
            )
        self._lang = cfg.lang

        # prepare fingerprint
        self._fingerprint = Fingerprint(
            features={
                "host": cfg.database_path,
                "index_name": cfg.index_name,
            }
        )
        return

    def _add_passages(self, passages: Iterable[dict[str, str]]):
        logger.warning(
            "bm25s Retriever does not support add passages. This function will build the index from scratch."
        )
        corpus = [i["text"] for i in passages]
        corpus_tokens = bm25s.tokenize(
            corpus, stopwords=self._lang, stemmer=self._stemmer
        )
        self._retriever.index(corpus_tokens)
        self._retriever.save(self.database_path, corpus=corpus)
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # retrieve
        search_method = search_kwargs.pop("search_method", "full_text")
        match search_method:
            case "full_text":
                contexts, scores = self._full_text_search(query, top_k, **search_kwargs)
            case _:
                raise ValueError(f"Unsupported search method: {search_method}")
        query_tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        contexts, scores = self._retriever.retrieve(
            query_tokens, k=top_k, show_progress=False, **search_kwargs
        )

        # form final results
        results = []
        for q, ctxs, score in zip(query, contexts, scores):
            results.append(
                [
                    RetrievedContext(
                        retriever=self.name,
                        query=q,
                        text=ctx["text"],
                        full_text=ctx["text"],
                        chunk_id=ctx["id"],
                        score=score[i],
                    )
                    for i, ctx in enumerate(ctxs)
                ]
            )
        return results

    def _full_text_search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        query_tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        return self._retriever.retrieve(
            query_tokens, k=top_k, show_progress=False, **search_kwargs
        )

    def clean(self) -> None:
        del self._retriever.scores
        del self._retriever.vocab_dict
        return

    def close(self) -> None:
        return

    def __len__(self) -> int:
        if hasattr(self._retriever, "scores"):
            return self._retriever.scores.get("num_docs", 0)
        return 0

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()
