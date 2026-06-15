from typing import Annotated, Any, Iterable, Optional

import bm25s
import numpy as np

from flexrag.common import LOGGER_MANAGER, Choices, configure

from .index_base import (
    DEFAULT_INDEX_BATCH_SIZE,
    RETRIEVER_INDEX,
    ContextIndexBase,
    IndexFieldsConfig,
    RawIndexBase,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.index.bm25")


@configure
class BM25RawIndexConfig:
    """Configuration for row-level BM25 indexing.

    :param method: BM25S scoring method. Available choices are ``"atire"``,
        ``"bm25l"``, ``"bm25+"``, ``"lucene"``, and ``"robertson"``. Defaults
        to ``"lucene"``.
    :param idf_method: BM25S IDF method. If ``None``, BM25S uses the default
        method for ``method``. Available choices are ``"atire"``, ``"bm25l"``,
        ``"bm25+"``, ``"lucene"``, and ``"robertson"``.
    :param backend: BM25S execution backend. Available choices are ``"numpy"``,
        ``"numba"``, and ``"auto"``. Defaults to ``"auto"``.
    :param k1: BM25 term-frequency saturation parameter. Defaults to 1.5.
    :param b: BM25 document-length normalization parameter. Defaults to 0.75.
    :param delta: Delta parameter used by BM25 variants that support it.
        Defaults to 0.5.
    :param lang: Language used for tokenization, stemming, and stopwords.
        Defaults to ``"english"``.
    :param show_progress: Whether BM25S should display progress while building
        the raw index. Defaults to ``True``.
    :param mmap: Whether to memory-map BM25S artifacts when loading from disk.
        Defaults to ``True``.
    """

    method: Annotated[
        str,
        Choices(
            "atire",
            "bm25l",
            "bm25+",
            "lucene",
            "robertson",
        ),
    ] = "lucene"
    idf_method: Optional[
        Annotated[
            str,
            Choices(
                "atire",
                "bm25l",
                "bm25+",
                "lucene",
                "robertson",
            ),
        ]
    ] = None
    backend: Annotated[str, Choices("numpy", "numba", "auto")] = "auto"
    k1: float = 1.5
    b: float = 0.75
    delta: float = 0.5
    lang: str = "english"
    show_progress: bool = True
    mmap: bool = True


class BM25RawIndex(RawIndexBase):
    """Row-level BM25 index backed by bm25s."""

    config_cls = BM25RawIndexConfig
    cfg: BM25RawIndexConfig

    def __init__(self, cfg: BM25RawIndexConfig) -> None:
        super().__init__(cfg)
        try:
            import Stemmer  # type: ignore

            self._stemmer = Stemmer.Stemmer(self.cfg.lang)
        except ImportError:
            logger.warning(
                "Stemmer is not available. "
                "You can install `PyStemmer` by `pip install PyStemmer` "
                "for better results."
            )
            self._stemmer = None

        self.index = bm25s.BM25(
            method=self.cfg.method,
            idf_method=self.cfg.idf_method,
            backend=self.cfg.backend,
            k1=self.cfg.k1,
            b=self.cfg.b,
            delta=self.cfg.delta,
        )
        return

    def build_index(
        self,
        data: Iterable[Any],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        logger.info("Preparing the passages for indexing.")
        items = list(data)

        logger.info("Building the index.")
        indexed_tokens = bm25s.tokenize(
            items,
            stopwords=self.cfg.lang,
            stemmer=self._stemmer,
            show_progress=self.cfg.show_progress,
            leave=True,
        )
        self.index.index(
            indexed_tokens,
            show_progress=self.cfg.show_progress,
            leave_progress=True,
        )
        return

    def insert(self, data: list[Any]) -> None:
        raise NotImplementedError("BM25RawIndex does not support inserting data.")

    def search(
        self,
        query: list[str],
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

    def save_to_local(self, index_path: str) -> None:
        self._save_config(index_path)
        self.index.save(index_path)
        return

    def _load_from_local(self, index_path: str) -> None:
        try:
            self.index = bm25s.BM25.load(index_path, mmap=self.cfg.mmap)
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load index from {index_path}") from exc
        return

    def clear(self) -> None:
        if hasattr(self.index, "scores"):
            del self.index.scores
        if hasattr(self.index, "vocab_dict"):
            del self.index.vocab_dict
        return

    def __len__(self) -> int:
        if hasattr(self.index, "scores"):
            return self.index.scores.get("num_docs", 0)
        return 0

    @property
    def infimum(self) -> float:
        return 0.0

    @property
    def supremum(self) -> float:
        return float("inf")


@configure
class BM25IndexConfig(BM25RawIndexConfig, IndexFieldsConfig):
    """Configuration for context-level BM25Index.

    :param indexed_fields: Context fields to index. If ``None``, all fields are
        indexed. Defaults to ``None``.
    :param merge_method: How to merge multiple field-level scores for the same
        context. Available choices are ``"max"``, ``"sum"``, ``"mean"``, and
        ``"concat"``. Defaults to ``"max"``.
    :param method: BM25S scoring method. Available choices are ``"atire"``,
        ``"bm25l"``, ``"bm25+"``, ``"lucene"``, and ``"robertson"``. Defaults
        to ``"lucene"``.
    :param idf_method: BM25S IDF method. If ``None``, BM25S uses the default
        method for ``method``.
    :param backend: BM25S execution backend. Available choices are ``"numpy"``,
        ``"numba"``, and ``"auto"``. Defaults to ``"auto"``.
    :param k1: BM25 term-frequency saturation parameter. Defaults to 1.5.
    :param b: BM25 document-length normalization parameter. Defaults to 0.75.
    :param delta: Delta parameter used by BM25 variants that support it.
        Defaults to 0.5.
    :param lang: Language used for tokenization, stemming, and stopwords.
        Defaults to ``"english"``.
    :param show_progress: Whether BM25S should display progress while building
        the raw index. Defaults to ``True``.
    :param mmap: Whether to memory-map BM25S artifacts when loading from disk.
        Defaults to ``True``.
    """


@RETRIEVER_INDEX("bm25", config_class=BM25IndexConfig)
class BM25Index(ContextIndexBase):
    """Context-level BM25 index."""

    raw_index_cls = BM25RawIndex
    raw_config_cls = BM25RawIndexConfig
    config_cls = BM25IndexConfig
