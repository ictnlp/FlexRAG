from __future__ import annotations

import os
import pickle
import shutil
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Iterable

import faiss
import numpy as np

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.models.encoders.encoder_base import EncoderProtocol

from ..view import RetrievalViewConfig
from .base import Hit, SyncCollectionBackendBase


@configure
class FaissBackendConfig:
    """Configuration for ``FaissBackend``.

    :param path: Directory for Faiss index and state artifacts. Required at
        construction time.
    :param view: Retrieval view configuration. Required for new artifacts; may be
        omitted when loading an artifact that already persists its view.
    :param distance_function: Distance metric, one of ``"IP"``, ``"L2"``, or
        ``"COS"``.
    :param factory_str: Optional Faiss factory string. ``None`` auto-selects.
    :param index_train_num: Number of rows sampled for training, or ``-1`` for
        all rows.
    :param search_options: Default Faiss search-time options.
    :param mmap: Whether to memory-map persisted Faiss index artifacts on load.
    """

    path: str | Path | None = None
    view: RetrievalViewConfig | None = None
    distance_function: Annotated[str, Choices("IP", "L2", "COS")] = "IP"
    factory_str: str | None = None
    index_train_num: int = -1
    search_options: dict[str, Any] | None = None
    mmap: bool = True

    def __post_init__(self) -> None:
        if self.distance_function not in {"IP", "L2", "COS"}:
            raise ValueError(f"Invalid distance_function: {self.distance_function}")
        if self.index_train_num != -1 and self.index_train_num <= 0:
            raise ValueError("index_train_num must be -1 or a positive integer.")
        return


class FaissBackend(SyncCollectionBackendBase):
    """Dense local backend powered by Faiss.

    The backend stores projected row embeddings and row-to-context mappings. It
    requires a context store for hydration and full rebuild backfill.
    """

    requires_context_store = True

    def __init__(
        self,
        config: FaissBackendConfig,
        *,
        query_encoder: EncoderProtocol,
        passage_encoder: EncoderProtocol | None = None,
    ) -> None:
        """Create or load a Faiss backend.

        :param config: Faiss backend configuration.
        :param query_encoder: Runtime encoder for query objects.
        :param passage_encoder: Optional encoder for projected context content.
        :raises ValueError: If path is missing or no view can be provided or
            loaded.
        """
        if config.path is None:
            raise ValueError("FaissBackendConfig.path must be provided.")
        super().__init__(config.view.to_view() if config.view is not None else None)
        self.config = config
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.index: faiss.Index | None = None
        self.row_context_ids: list[str] = []
        self.max_field_num = 1
        self.resolved_factory_str: str | None = None
        self.rows_during_train = 0
        self.added_rows_since_train = 0
        self._load_if_present()
        self._require_view()
        return

    @property
    def is_addable(self) -> bool:
        """Return whether the loaded Faiss index can accept incremental rows."""
        return self.index is not None and bool(self.index.is_trained)

    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Rebuild the Faiss index from a complete corpus.

        :param contexts: Complete context corpus to project, encode, and index.
        """
        contents, row_context_ids, max_field_num = self._project_contexts(contexts)
        self.row_context_ids = row_context_ids
        self.max_field_num = max_field_num
        self.rows_during_train = len(row_context_ids)
        self.added_rows_since_train = 0
        if contents:
            embeddings = self.passage_encoder.encode(contents)
            embeddings = self._normalize_for_metric(embeddings)
            self.resolved_factory_str = self._resolve_factory_str(
                embedding_size=embeddings.shape[1],
                embedding_length=embeddings.shape[0],
                factory_str=self.config.factory_str,
            )
            self.index = self._prepare_index(
                distance_function=self.config.distance_function,
                embedding_size=embeddings.shape[1],
                factory_str=self.resolved_factory_str,
            )
            self._train_index(embeddings)
            self.index.add(embeddings)
        else:
            self.index = None
            self.resolved_factory_str = None
            self.rows_during_train = 0
        self._save_state()
        return

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Incrementally append projected rows to a trained Faiss index.

        :param contexts: New contexts to project and encode.
        :raises NotImplementedError: If the index is not currently addable.
        :raises ValueError: If encoded vectors do not match the index dimension.
        """
        if not self.is_addable:
            super().add_contexts(contexts)
        contents, row_context_ids, max_field_num = self._project_contexts(contexts)
        if not contents:
            return
        embeddings = self.passage_encoder.encode(contents)
        embeddings = self._normalize_for_metric(embeddings)
        assert self.index is not None
        if embeddings.shape[1] != self.index.d:
            raise ValueError(
                "Encoded embedding size does not match the existing Faiss index."
            )
        self.index.add(embeddings)
        self.row_context_ids.extend(row_context_ids)
        self.max_field_num = max(self.max_field_num, max_field_num)
        self.added_rows_since_train += len(row_context_ids)
        self._warn_if_rebuild_is_due()
        self._save_state()
        return

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        if not queries:
            return []
        if self.index is None or not self.row_context_ids or top_k <= 0:
            return [[] for _ in queries]
        view = self._require_view()
        query_vectors = self.query_encoder.encode(queries)
        query_vectors = self._normalize_for_metric(query_vectors)
        row_k = min(len(self.row_context_ids), top_k * self.max_field_num)
        search_params = self._prepare_search_params(search_options=search_options)
        if search_params is None:
            scores, indices = self.index.search(query_vectors, row_k)
        else:
            scores, indices = self.index.search(
                query_vectors,
                row_k,
                params=search_params,
            )
        scores = self._postprocess_scores(scores)
        results = []
        for query_indices, query_scores in zip(indices, scores):
            by_context: dict[str, list[float]] = defaultdict(list)
            for row_idx, score in zip(query_indices, query_scores):
                idx = int(row_idx)
                if idx < 0 or idx >= len(self.row_context_ids):
                    continue
                by_context[self.row_context_ids[idx]].append(float(score))
            ordered = sorted(
                (
                    (context_id, view.aggregate_scores(context_scores))
                    for context_id, context_scores in by_context.items()
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            results.append(
                [
                    Hit(
                        context_id=context_id,
                        score=score,
                        backend="",
                        view=view.name,
                    )
                    for context_id, score in ordered[:top_k]
                ]
            )
        return results

    def clear(self) -> None:
        """Clear in-memory state and delete backend artifacts."""
        self.index = None
        self.row_context_ids = []
        self.max_field_num = 1
        self.resolved_factory_str = None
        self.rows_during_train = 0
        self.added_rows_since_train = 0
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        return

    def count(self) -> int:
        return len(set(self.row_context_ids))

    def close(self) -> None:
        return

    def _project_contexts(
        self,
        contexts: Iterable[Context],
    ) -> tuple[list[Any], list[str], int]:
        contents = []
        row_context_ids = []
        max_field_num = 1
        view = self._require_view()
        for context in contexts:
            projected = view.project(context)
            if not projected:
                continue
            max_field_num = max(max_field_num, len(projected))
            for row in projected:
                contents.append(row.content)
                row_context_ids.append(row.context_id)
        return contents, row_context_ids, max_field_num

    def _normalize_for_metric(self, vectors: np.ndarray) -> np.ndarray:
        if self.config.distance_function != "COS" or vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-12, a_max=None)
        return vectors / norms

    def _postprocess_scores(self, scores: np.ndarray) -> np.ndarray:
        if self.config.distance_function == "L2":
            return -scores
        return scores

    def _resolve_factory_str(
        self,
        *,
        embedding_size: int,
        embedding_length: int,
        factory_str: str | None,
    ) -> str:
        if factory_str is not None:
            return factory_str

        n = embedding_length
        d = embedding_size
        raw_bytes = n * d * np.dtype(np.float32).itemsize
        n_list = max(64, 2 ** int(np.log2(np.sqrt(n))))

        if n < 10_000 or (n * d <= 20_000_000 and raw_bytes <= 128 * 1024 * 1024):
            return "Flat"

        if d <= 1024 and n <= 300_000:
            if d <= 256:
                hnsw_m = 32
            elif d <= 768:
                hnsw_m = 24
            else:
                hnsw_m = 16
            return f"HNSW{hnsw_m}"

        if n <= 1_000_000 and raw_bytes <= 8 * 1024 * 1024 * 1024:
            return f"IVF{n_list},Flat"

        pq_m = None
        for m in range(min(64, d // 2), 7, -1):
            if d % m == 0:
                pq_m = m
                break
        if pq_m is None:
            resolved = f"IVF{n_list},Flat"
            warnings.warn(
                "Unable to derive a suitable PQ configuration for embedding size "
                f"{d}. Falling back to {resolved}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return resolved

        return f"IVF{n_list},PQ{pq_m}x4fs"

    def _prepare_index(
        self,
        *,
        distance_function: str,
        embedding_size: int,
        factory_str: str,
    ) -> faiss.Index:
        match distance_function:
            case "IP" | "COS":
                basic_metric = faiss.METRIC_INNER_PRODUCT
            case "L2":
                basic_metric = faiss.METRIC_L2
            case _:
                raise ValueError(f"Invalid distance_function: {distance_function}")
        return faiss.index_factory(embedding_size, factory_str, basic_metric)

    def _train_index(self, embeddings: np.ndarray) -> None:
        assert self.index is not None
        if self.index.is_trained:
            return
        if (
            self.config.index_train_num >= embeddings.shape[0]
            or self.config.index_train_num == -1
        ):
            self.index.train(embeddings)
            return
        selected_indices = np.random.choice(
            embeddings.shape[0],
            self.config.index_train_num,
            replace=False,
        )
        selected_indices = np.sort(selected_indices)
        self.index.train(embeddings[selected_indices])
        return

    def _prepare_search_params(
        self,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> Any | None:
        if self.index is None:
            return None
        options = dict(self.config.search_options or {})
        options.update(search_options or {})
        k_factor = options.get("k_factor", 10)
        n_probe = options.get("n_probe")
        if n_probe is None:
            n_probe = getattr(self.index, "nlist", 256) // 8
        polysemous_ht = options.get("polysemous_ht", 0)
        ef_search = options.get("efSearch", 100)

        def get_search_params(index: faiss.Index) -> Any | None:
            if isinstance(index, faiss.IndexRefine):
                return faiss.IndexRefineSearchParameters(
                    k_factor=k_factor,  # type: ignore
                    base_index_params=get_search_params(
                        faiss.downcast_index(index.base_index)
                    ),  # type: ignore
                )
            if isinstance(index, faiss.IndexPreTransform):
                return faiss.SearchParametersPreTransform(
                    index_params=get_search_params(  # type: ignore
                        faiss.downcast_index(index.index)
                    )
                )
            if isinstance(index, faiss.IndexIVFPQ):
                if hasattr(index, "quantizer"):
                    return faiss.IVFPQSearchParameters(
                        nprobe=n_probe,
                        polysemous_ht=polysemous_ht,
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),
                    )
                return faiss.IVFPQSearchParameters(
                    nprobe=n_probe,
                    polysemous_ht=polysemous_ht,
                )
            if isinstance(index, faiss.IndexIVF):
                if hasattr(index, "quantizer"):
                    return faiss.SearchParametersIVF(
                        nprobe=n_probe,  # type: ignore
                        quantizer_params=get_search_params(
                            faiss.downcast_index(index.quantizer)
                        ),  # type: ignore
                    )
                return faiss.SearchParametersIVF(nprobe=n_probe)  # type: ignore
            if isinstance(index, faiss.IndexHNSW):
                return faiss.SearchParametersHNSW(efSearch=ef_search)
            if isinstance(index, faiss.IndexPQ):
                return faiss.SearchParametersPQ(polysemous_ht=polysemous_ht)
            return None

        return get_search_params(faiss.downcast_index(self.index))

    def _warn_if_rebuild_is_due(self) -> None:
        if self.resolved_factory_str == "Flat":
            return
        if self.rows_during_train <= 0:
            return
        if self.added_rows_since_train < self.rows_during_train:
            return
        warnings.warn(
            "FaissBackend has added at least as many rows as it had during the "
            "last rebuild; consider rebuilding the index.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    @property
    def _index_path(self) -> Path:
        return self.path / "index.faiss"

    @property
    def _state_path(self) -> Path:
        return self.path / "state.pkl"

    def _save_state(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, os.fspath(self._index_path))
        elif self._index_path.exists():
            self._index_path.unlink()
        with open(self._state_path, "wb") as f:
            pickle.dump(
                {
                    "view": self._require_view().to_dict(),
                    "row_context_ids": self.row_context_ids,
                    "max_field_num": self.max_field_num,
                    "resolved_factory_str": self.resolved_factory_str,
                    "rows_during_train": self.rows_during_train,
                    "added_rows_since_train": self.added_rows_since_train,
                },
                f,
            )
        return

    def _load_if_present(self) -> None:
        if not self._state_path.exists():
            return
        with open(self._state_path, "rb") as f:
            state = pickle.load(f)
        self._load_persisted_view(state.get("view"))
        self.row_context_ids = list(state.get("row_context_ids", []))
        self.max_field_num = int(state.get("max_field_num", 1))
        self.resolved_factory_str = state.get(
            "resolved_factory_str",
            self.config.factory_str or "Flat",
        )
        self.rows_during_train = int(
            state.get("rows_during_train", len(self.row_context_ids))
        )
        self.added_rows_since_train = int(state.get("added_rows_since_train", 0))
        if self.row_context_ids and self._index_path.exists():
            if self.config.mmap:
                self.index = faiss.read_index(
                    os.fspath(self._index_path),
                    faiss.IO_FLAG_MMAP,
                )
            else:
                self.index = faiss.read_index(os.fspath(self._index_path))
        return
