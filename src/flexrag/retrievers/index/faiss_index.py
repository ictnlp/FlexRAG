import os
from typing import Any, Iterable, Optional

import faiss
import numpy as np

from flexrag.common import LOGGER_MANAGER, configure
from flexrag.models import EncoderProtocol

from .index_base import (
    DEFAULT_INDEX_BATCH_SIZE,
    RETRIEVER_INDEX,
    ContextIndexBase,
    DenseRawIndexBase,
    DenseRawIndexBaseConfig,
    IndexFieldsConfig,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retriever.index.faiss")


@configure
class FaissRawIndexConfig(DenseRawIndexBaseConfig):
    """Configuration for row-level Faiss indexing.

    :param distance_function: Vector distance or similarity function. Available
        choices are ``"IP"``, ``"L2"``, and ``"COS"``. Defaults to ``"IP"``.
    :param factory_str: Optional Faiss factory string. If ``None``, FlexRAG
        selects a factory string based on embedding size and corpus size.
    :param index_train_num: Number of embeddings sampled to train trainable
        Faiss indexes. ``-1`` means all embeddings. Defaults to -1.
    :param n_probe: Number of IVF cells to probe during search. If ``None``,
        a default is derived from the index when possible.
    :param k_factor: Refinement factor used for Faiss refine indexes. Defaults
        to 10.
    :param polysemous_ht: Polysemous hash threshold used by compatible Faiss
        indexes. Defaults to 0.
    :param efSearch: HNSW search effort used by compatible Faiss indexes.
        Defaults to 100.
    """

    factory_str: Optional[str] = None
    index_train_num: int = -1
    n_probe: Optional[int] = None
    k_factor: int = 10
    polysemous_ht: int = 0
    efSearch: int = 100


class FaissRawIndex(DenseRawIndexBase):
    """Row-level Faiss index."""

    config_cls = FaissRawIndexConfig
    cfg: FaissRawIndexConfig

    def __init__(
        self,
        cfg: FaissRawIndexConfig,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
    ) -> None:
        super().__init__(cfg, query_encoder, passage_encoder)
        self.index = None
        return

    def build_index(
        self,
        data: Iterable[Any],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        self.clear()
        embeddings = self.encode_data_batch(
            data,
            is_query=False,
            batch_size=batch_size,
            scratch_path=scratch_path,
        )
        factory_str = self._resolve_factory_str(
            embedding_size=embeddings.shape[1],
            embedding_length=embeddings.shape[0],
            factory_str=self.cfg.factory_str,
        )
        self.index = self._prepare_index(
            distance_function=self.cfg.distance_function,
            embedding_size=embeddings.shape[1],
            factory_str=factory_str,
        )
        self._train_index(embeddings)
        self.add_embeddings_batch(embeddings, batch_size=batch_size)
        if isinstance(embeddings, np.memmap):
            emb_path = embeddings.filename
            os.remove(emb_path)
        return

    def _resolve_factory_str(
        self,
        embedding_size: int,
        embedding_length: int,
        factory_str: Optional[str] = None,
    ) -> str:
        if factory_str is not None:
            logger.info(f"Using Faiss factory string: {factory_str}")
            return factory_str

        n = embedding_length
        d = embedding_size
        raw_bytes = n * d * np.dtype(np.float32).itemsize
        n_list = max(64, 2 ** int(np.log2(np.sqrt(n))))

        if n < 10_000 or (n * d <= 20_000_000 and raw_bytes <= 128 * 1024 * 1024):
            resolved_factory_str = "Flat"
            logger.info("Auto set index to Flat")
            return resolved_factory_str

        if d <= 1024 and n <= 300_000:
            if d <= 256:
                hnsw_m = 32
            elif d <= 768:
                hnsw_m = 24
            else:
                hnsw_m = 16
            resolved_factory_str = f"HNSW{hnsw_m}"
            logger.info(f"Auto set index to {resolved_factory_str}")
            return resolved_factory_str

        if n <= 1_000_000 and raw_bytes <= 8 * 1024 * 1024 * 1024:
            resolved_factory_str = f"IVF{n_list},Flat"
            logger.info(f"Auto set index to {resolved_factory_str}")
            logger.info(
                f"We recommend to set n_probe to {n_list // 8} "
                "for better inference performance."
            )
            return resolved_factory_str

        pq_m = None
        for m in range(min(64, d // 2), 7, -1):
            if d % m == 0:
                pq_m = m
                break
        if pq_m is None:
            resolved_factory_str = f"IVF{n_list},Flat"
            logger.warning(
                "Unable to derive a suitable PQ configuration for embedding size "
                f"{d}. Falling back to {resolved_factory_str}."
            )
            logger.info(
                f"We recommend to set n_probe to {n_list // 8} "
                "for better inference performance."
            )
            return resolved_factory_str

        resolved_factory_str = f"IVF{n_list},PQ{pq_m}x4fs"
        logger.info(f"Auto set index to {resolved_factory_str}")
        logger.info(
            f"We recommend to set n_probe to {n_list // 8} "
            "for better inference performance."
        )
        return resolved_factory_str

    def _prepare_index(
        self,
        distance_function: str,
        embedding_size: int,
        factory_str: str,
    ):
        match distance_function:
            case "IP" | "COS":
                basic_metric = faiss.METRIC_INNER_PRODUCT
            case "L2":
                basic_metric = faiss.METRIC_L2
            case _:
                raise ValueError(f"Unknown distance function: {distance_function}")

        return faiss.index_factory(embedding_size, factory_str, basic_metric)

    def _train_index(self, embeddings: np.ndarray) -> None:
        if self.is_trained:
            logger.info("Index is trained already.")
            return
        logger.info("Training index")
        if (self.cfg.index_train_num >= embeddings.shape[0]) or (
            self.cfg.index_train_num == -1
        ):
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype("float32")
            self.index.train(embeddings)
        else:
            selected_indices = np.random.choice(
                embeddings.shape[0],
                self.cfg.index_train_num,
                replace=False,
            )
            selected_indices = np.sort(selected_indices)
            selected_embeddings = embeddings[selected_indices].astype("float32")
            self.index.train(selected_embeddings)
        return

    def add_embeddings(self, embeddings: np.ndarray) -> None:
        embeddings = embeddings.astype("float32")
        assert self.is_trained, "Index should be trained first"
        self.index.add(embeddings)
        return

    def _prepare_search_params(self, **kwargs):
        k_factor = kwargs.get("k_factor", self.cfg.k_factor)
        n_probe = kwargs.get("n_probe", self.cfg.n_probe)
        if n_probe is None:
            n_probe = getattr(self.index, "nlist", 256) // 8
        polysemous_ht = kwargs.get("polysemous_ht", self.cfg.polysemous_ht)
        efSearch = kwargs.get("efSearch", self.cfg.efSearch)

        def get_search_params(index):
            if isinstance(index, faiss.IndexRefine):
                params = faiss.IndexRefineSearchParameters(
                    k_factor=k_factor,  # type: ignore
                    base_index_params=get_search_params(
                        faiss.downcast_index(index.base_index)
                    ),  # type: ignore
                )
            elif isinstance(index, faiss.IndexPreTransform):
                params = faiss.SearchParametersPreTransform(
                    index_params=get_search_params(  # type: ignore
                        faiss.downcast_index(index.index)
                    )
                )
            elif isinstance(index, faiss.IndexIVFPQ):
                if hasattr(index, "quantizer"):
                    params = faiss.IVFPQSearchParameters(
                        nprobe=n_probe,
                        polysemous_ht=polysemous_ht,
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),
                    )
                else:
                    params = faiss.IVFPQSearchParameters(
                        nprobe=n_probe, polysemous_ht=polysemous_ht
                    )
            elif isinstance(index, faiss.IndexIVF):
                if hasattr(index, "quantizer"):
                    params = faiss.SearchParametersIVF(
                        nprobe=n_probe,  # type: ignore
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),  # type: ignore
                    )
                else:
                    params = faiss.SearchParametersIVF(
                        nprobe=n_probe  # type: ignore
                    )
            elif isinstance(index, faiss.IndexHNSW):
                params = faiss.SearchParametersHNSW(efSearch=efSearch)
            elif isinstance(index, faiss.IndexPQ):
                params = faiss.SearchParametersPQ(polysemous_ht=polysemous_ht)
            else:
                params = None
            return params

        return get_search_params(self.index)

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        query_vectors = self.encode_data(query, is_query=True)
        search_params = self._prepare_search_params(**search_kwargs)
        scores, indices = self.index.search(query_vectors, top_k, params=search_params)
        scores = self._postprocess_scores(scores)
        return indices, scores

    def save_to_local(self, index_path: str) -> None:
        assert self.index is not None and self.index.is_trained, (
            "Index should be trained first."
        )
        self._save_config(index_path)
        faiss.write_index(self.index, os.path.join(index_path, "index.faiss"))
        return

    def _load_from_local(self, index_path: str) -> None:
        try:
            self.index = faiss.read_index(
                os.path.join(index_path, "index.faiss"),
                faiss.IO_FLAG_MMAP,
            )
        except Exception as exc:
            raise FileNotFoundError(f"Unable to load index from {index_path}") from exc
        return

    def clear(self) -> None:
        if self.index is None:
            return
        self.index.reset()
        return

    @property
    def embedding_size(self) -> int:
        if self.index is not None:
            return self.index.d
        if self.passage_encoder is not None:
            return self.passage_encoder.embedding_size
        if self.query_encoder is not None:
            return self.query_encoder.embedding_size
        raise ValueError("Index is not initialized.")

    @property
    def is_trained(self) -> bool:
        if self.index is None:
            return False
        return self.index.is_trained

    @property
    def is_addable(self) -> bool:
        return self.is_trained

    def __len__(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal


@configure
class FaissIndexConfig(FaissRawIndexConfig, IndexFieldsConfig):
    """Configuration for context-level FaissIndex.

    :param indexed_fields: Context fields to index. If ``None``, all fields are
        indexed. Defaults to ``None``.
    :param merge_method: How to merge multiple field-level scores for the same
        context. Available choices are ``"max"``, ``"sum"``, ``"mean"``, and
        ``"concat"``. Defaults to ``"max"``.
    :param distance_function: Vector distance or similarity function. Available
        choices are ``"IP"``, ``"L2"``, and ``"COS"``. Defaults to ``"IP"``.
    :param factory_str: Optional Faiss factory string. If ``None``, FlexRAG
        selects a factory string based on embedding size and corpus size.
    :param index_train_num: Number of embeddings sampled to train trainable
        Faiss indexes. ``-1`` means all embeddings. Defaults to -1.
    :param n_probe: Number of IVF cells to probe during search. If ``None``,
        a default is derived from the index when possible.
    :param k_factor: Refinement factor used for Faiss refine indexes. Defaults
        to 10.
    :param polysemous_ht: Polysemous hash threshold used by compatible Faiss
        indexes. Defaults to 0.
    :param efSearch: HNSW search effort used by compatible Faiss indexes.
        Defaults to 100.
    """


@RETRIEVER_INDEX("faiss", config_class=FaissIndexConfig)
class FaissIndex(ContextIndexBase):
    """Context-level Faiss index."""

    raw_index_cls = FaissRawIndex
    raw_config_cls = FaissRawIndexConfig
    config_cls = FaissIndexConfig

    @property
    def query_encoder(self):
        return self.raw_index.query_encoder

    @property
    def passage_encoder(self):
        return self.raw_index.passage_encoder
