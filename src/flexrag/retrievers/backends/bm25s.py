from __future__ import annotations

import os
import pickle
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Iterable

import bm25s

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context

from ..view import RetrievalViewConfig
from .base import Hit, SyncCollectionBackendBase


@configure
class BM25SBackendConfig:
    """Configuration for ``BM25SBackend``.

    :param path: Directory for BM25S artifacts and state. Required at
        construction time.
    :param view: Retrieval view configuration. Required for new artifacts; may be
        omitted when loading an artifact that already persists its view.
    :param method: BM25 scoring method passed to ``bm25s.BM25``.
    :param idf_method: Optional IDF method passed to ``bm25s.BM25``.
    :param backend: BM25S compute backend.
    :param lang: Language used for stopwords and optional stemming.
    :param show_progress: Whether BM25S should display progress bars.
    :param k1: BM25 ``k1`` parameter.
    :param b: BM25 ``b`` parameter.
    :param delta: BM25 ``delta`` parameter.
    :param mmap: Whether to memory-map persisted BM25S artifacts on load.
    """

    path: str | Path | None = None
    view: RetrievalViewConfig | None = None
    method: Annotated[
        str,
        Choices("atire", "bm25l", "bm25+", "lucene", "robertson"),
    ] = "lucene"
    idf_method: Annotated[
        str,
        Choices("atire", "bm25l", "bm25+", "lucene", "robertson"),
    ] | None = None
    backend: Annotated[str, Choices("numpy", "numba", "auto")] = "auto"
    lang: str = "english"
    show_progress: bool = False
    k1: float = 1.5
    b: float = 0.75
    delta: float = 0.5
    mmap: bool = True


class BM25SBackend(SyncCollectionBackendBase):
    """Text sparse backend powered by BM25S.

    BM25S stores only projected text rows and therefore requires a context store
    for hydration and rebuild backfill.
    """

    requires_context_store = True
    is_addable = False

    def __init__(
        self,
        config: BM25SBackendConfig,
    ) -> None:
        """Create or load a BM25S backend.

        :param config: BM25S backend configuration.
        :raises ValueError: If path is missing or no view can be provided or
            loaded.
        """
        if config.path is None:
            raise ValueError("BM25SBackendConfig.path must be provided.")
        super().__init__(config.view.to_view() if config.view is not None else None)
        self.config = config
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._stemmer = self._make_stemmer()
        self.index = self._make_index()
        self.row_context_ids: list[str] = []
        self.max_field_num = 1
        self._load_if_present()
        self._require_view()
        return

    def rebuild(self, contexts: Iterable[Context]) -> None:
        """Rebuild the BM25S index from a complete corpus.

        :param contexts: Complete context corpus to project and index.
        :raises TypeError: If any projected content is not text.
        """
        view = self._require_view()
        rows = []
        row_context_ids = []
        max_field_num = 1
        for context in contexts:
            projected = view.project(context)
            if not projected:
                continue
            max_field_num = max(max_field_num, len(projected))
            for row in projected:
                if not isinstance(row.content, str):
                    raise TypeError(
                        "BM25SBackend only supports text retrieval content; "
                        f"field {row.field!r} has type {type(row.content).__name__}."
                    )
                rows.append(row.content)
                row_context_ids.append(row.context_id)

        self.index = self._make_index()
        self.row_context_ids = row_context_ids
        self.max_field_num = max_field_num
        if rows:
            tokens = bm25s.tokenize(
                rows,
                stopwords=self.config.lang,
                stemmer=self._stemmer,
                show_progress=self.config.show_progress,
            )
            self.index.index(
                tokens,
                show_progress=self.config.show_progress,
            )
        self._save_state()
        return

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        view = self._require_view()
        if not self.row_context_ids or top_k <= 0:
            return [[] for _ in queries]
        if any(not isinstance(query, str) for query in queries):
            raise TypeError("BM25SBackend only supports text queries.")
        query_tokens = bm25s.tokenize(
            queries,
            stemmer=self._stemmer,
            show_progress=False,
        )
        row_k = min(len(self.row_context_ids), top_k * self.max_field_num)
        row_indices, row_scores = self.index.retrieve(
            query_tokens,
            k=row_k,
            show_progress=False,
        )
        results = []
        for indices, scores in zip(row_indices, row_scores):
            by_context: dict[str, list[float]] = defaultdict(list)
            for row_idx, score in zip(indices, scores):
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
        self.index = self._make_index()
        self.row_context_ids = []
        self.max_field_num = 1
        if self.path.exists():
            shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)
        return

    def count(self) -> int:
        return len(set(self.row_context_ids))

    def close(self) -> None:
        return

    def _make_index(self) -> Any:
        return bm25s.BM25(
            method=self.config.method,
            idf_method=self.config.idf_method,
            backend=self.config.backend,
            k1=self.config.k1,
            b=self.config.b,
            delta=self.config.delta,
        )

    def _make_stemmer(self) -> Any | None:
        try:
            import Stemmer  # type: ignore

            return Stemmer.Stemmer(self.config.lang)
        except ImportError:
            return None

    @property
    def _index_path(self) -> Path:
        return self.path / "bm25s"

    @property
    def _state_path(self) -> Path:
        return self.path / "state.pkl"

    def _save_state(self) -> None:
        if self._index_path.exists():
            shutil.rmtree(self._index_path)
        if self.row_context_ids:
            self.index.save(os.fspath(self._index_path))
        with open(self._state_path, "wb") as f:
            pickle.dump(
                {
                    "view": self._require_view().to_dict(),
                    "row_context_ids": self.row_context_ids,
                    "max_field_num": self.max_field_num,
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
        if self.row_context_ids and self._index_path.exists():
            self.index = bm25s.BM25.load(
                os.fspath(self._index_path),
                mmap=self.config.mmap,
            )
        return
