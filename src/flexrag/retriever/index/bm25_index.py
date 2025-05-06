import os
import shutil
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import bm25s
import numpy as np
from omegaconf import OmegaConf

from flexrag.utils import LOGGER_MANAGER, Choices

from .index_base import RETRIEVER_INDEX, RetrieverIndexBase, RetrieverIndexBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.bm25")


@dataclass
class BM25IndexConfig(RetrieverIndexBaseConfig):
    """Configuration class for BM25Index.

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
    """

    method: Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"]) = "lucene"  # type: ignore
    idf_method: Optional[Choices(["atire", "bm25l", "bm25+", "lucene", "robertson"])] = None  # type: ignore
    backend: Choices(["numpy", "numba", "auto"]) = "auto"  # type: ignore
    k1: float = 1.5
    b: float = 0.75
    delta: float = 0.5
    lang: str = "english"


@RETRIEVER_INDEX("bm25", config_class=BM25IndexConfig)
class BM25Index(RetrieverIndexBase):
    """BM25Index is a index that retrieves passages using the BM25 algorithm.
    The implementation is based on the `bm25s <https://github.com/xhluca/bm25s>`_ project.
    """

    def __init__(self, cfg: BM25IndexConfig) -> None:
        self.cfg = BM25IndexConfig.extract(cfg)
        try:
            import Stemmer

            self._stemmer = Stemmer.Stemmer(cfg.lang)
        except:
            logger.warning(
                "Stemmer is not available. "
                "You can install `PyStemmer` by `pip install PyStemmer` for better results."
            )
            self._stemmer = None

        # initialize the index
        self.index = bm25s.BM25(
            method=cfg.method,
            idf_method=cfg.idf_method,
            backend=cfg.backend,
            k1=cfg.k1,
            b=cfg.b,
            delta=cfg.delta,
        )

        # load the index if index_path is provided
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                logger.info(f"Loading index from {self.cfg.index_path}")
                try:
                    self.index = bm25s.BM25.load(self.cfg.index_path, mmap=True)
                except:
                    raise FileNotFoundError(
                        f"Unable to load index from {self.cfg.index_path}"
                    )
        return

    def build_index(self, data: Iterable[Any]) -> None:
        # prepare the data
        logger.info("Preparing the passages for indexing.")
        items = list(data)

        # tokenize and build index
        logger.info("Building the index.")
        indexed_tokens = bm25s.tokenize(
            items, stopwords=self.cfg.lang, stemmer=self._stemmer
        )
        self.index.index(indexed_tokens)

        # serialize index
        if self.cfg.index_path is not None:
            self.save_to_local()
        return

    def insert(self, data: list[Any]) -> None:
        raise NotImplementedError(
            "BM25Index currently does not support inserting data."
        )

    def search(
        self,
        query: list[str],  # bm25 algorithm can only process string
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_tokens = bm25s.tokenize(query, stemmer=self._stemmer, show_progress=False)
        contexts, scores = self.index.retrieve(
            query_tokens,
            k=top_k,
            show_progress=False,
            **search_kwargs,
        )
        return contexts, scores

    @property
    def is_addable(self) -> bool:
        return False

    def save_to_local(self, index_path: str = None) -> None:
        # check if the index is serializable
        if index_path is not None:
            self.cfg.index_path = index_path
        assert self.cfg.index_path is not None, "`index_path` is not set."
        if not os.path.exists(index_path):
            os.makedirs(self.cfg.index_path)
        logger.info(f"Serializing index to {self.cfg.index_path}")

        # save the configuration
        config_path = os.path.join(self.cfg.index_path, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            OmegaConf.save(self.cfg, f)
        id_path = os.path.join(self.cfg.index_path, "cls.id")
        with open(id_path, "w", encoding="utf-8") as f:
            f.write(self.__class__.__name__)

        # serialize the index
        self.index.save(self.cfg.index_path)
        return

    def clear(self) -> None:
        del self.index.scores
        del self.index.vocab_dict
        if self.cfg.index_path is not None:
            if os.path.exists(self.cfg.index_path):
                shutil.rmtree(self.cfg.index_path)
        return

    def __len__(self) -> int:
        if hasattr(self.index, "scores"):
            return self.index.scores.get("num_docs", 0)
        return 0
