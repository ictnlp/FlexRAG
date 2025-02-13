import os
from dataclasses import dataclass
from functools import cached_property
from hashlib import sha1
from typing import Iterable, Optional

import bm25s
from omegaconf import MISSING

from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.utils import LOGGER_MANAGER, TIME_METER, Choices, SimpleProgressLogger

from .retriever_base import RETRIEVERS, EditableRetriever, EditableRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.bm25s")


@dataclass
class BM25SRetrieverConfig(EditableRetrieverConfig):
    """Configuration class for BM25SRetriever.

    :param database_path: Path to the database directory. Required.
    :type database_path: str
    :param method: BM25S method. Default: "lucene".
        Available options: "atire", "bm25l", "bm25+", "lucene", "robertson".
    :type method: str
    :param idf_method: IDF method. Default: None.
        Available options: "atire", "bm25l", "bm25+", "lucene", "robertson".
    :type idf_method: Optional[str]
    :param backend: Backend for BM25S. Default: "auto".
        Available options: "numpy", "numba", "auto".
    :type backend: str
    :param k1: BM25S parameter k1. Default: 1.5.
    :type k1: float
    :param b: BM25S parameter b. Default: 0.75.
    :type b: float
    :param delta: BM25S parameter delta. Default: 0.5.
    :type delta: float
    :param lang: Language for Tokenization. Default: "english".
    :type lang: str
    :param indexed_fields: Fields to be indexed. None stands for all fields. Default: None.
    :type indexed_fields: Optional[list[str]]
    """

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
class BM25SRetriever(EditableRetriever):
    name = "BM25SSearch"

    def __init__(self, cfg: BM25SRetrieverConfig) -> None:
        super().__init__(cfg)
        # set basic args
        try:
            import Stemmer

            self._stemmer = Stemmer.Stemmer(cfg.lang)
        except:
            logger.warning(
                "Stemmer is not available. "
                "You can install `PyStemmer` by `pip install PyStemmer` for better results."
            )
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
    def add_passages(self, passages: Iterable[Context]):
        logger.warning(
            "bm25s Retriever does not support add passages. This function will build the index from scratch."
        )
        logger.info("Preparing the passages for indexing.")

        # prepare the passages
        p_logger = SimpleProgressLogger(logger, interval=self.log_interval)
        passages_ = []
        indexed_ = []
        for p in passages:
            data = p.data.copy()
            data[self.id_field_name] = p.context_id
            passages_.append(data)
            if len(self._indexed_fields) == 1:
                indexed_.append(data[self._indexed_fields[0]])
            else:
                indexed_.append(" ".join([data[f] for f in self._indexed_fields]))
            p_logger.update(1, "Preparing the passages")

        # tokenize and index
        logger.info("Indexing the passages.")
        indexed_tokens = bm25s.tokenize(
            indexed_, stopwords=self._lang, stemmer=self._stemmer
        )
        self._retriever.index(indexed_tokens)
        self._retriever.corpus = passages_
        self._retriever.save(self.database_path, corpus=passages_)
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
        results: list[list[RetrievedContext]] = []
        for q, ctxs, score in zip(query, contexts, scores):
            results.append([])
            for ctx, s in zip(ctxs, score):
                docid = ctx.pop(self.id_field_name)
                results[-1].append(
                    RetrievedContext(
                        context_id=docid,
                        retriever=self.name,
                        query=q,
                        data=ctx,
                        score=float(s),
                    )
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

    @cached_property
    def id_field_name(self) -> str:
        return sha1("context_id".encode()).hexdigest()
