from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from flexrag.common import Context
from flexrag.retrievers import (
    BM25SBackend,
    FaissBackend,
    FlexRetriever,
    LMDBContextStore,
    LMDBContextStoreConfig,
    RetrievalView,
)


class TokenEncoder:
    def encode(self, items: list[Any]) -> np.ndarray:
        keys = ("alpha", "beta", "gamma", "delta")
        return np.asarray(
            [[float(key in str(item).lower()) for key in keys] for item in items],
            dtype="float32",
        )


def contexts() -> list[Context]:
    return [
        Context(
            context_id="doc-alpha",
            data={"title": "alpha", "text": "alpha dense vector search"},
            source="memory",
        ),
        Context(
            context_id="doc-beta",
            data={"title": "beta", "text": "beta lexical keyword search"},
        ),
        Context(context_id="doc-gamma", data={"title": "gamma", "text": "gamma"}),
    ]


def test_bm25s_and_faiss_local_backend_round_trip(tmp_path: Path) -> None:
    store = LMDBContextStore(LMDBContextStoreConfig(path=tmp_path / "store"))
    text_view = RetrievalView("text", ["title", "text"])
    dense_view = RetrievalView("dense", ["text"])
    bm25s = BM25SBackend(text_view, tmp_path / "bm25s")
    faiss = FaissBackend(dense_view, tmp_path / "faiss", query_encoder=TokenEncoder())
    retriever = FlexRetriever.from_backends(
        {"bm25s": bm25s, "faiss": faiss},
        context_store=store,
    )
    retriever.add_contexts(contexts())
    assert retriever.count() == bm25s.count() == faiss.count() == 3
    assert bm25s.is_addable is False
    assert faiss.is_addable is True
    assert faiss.resolved_factory_str == "Flat"
    assert retriever.search("beta", top_k=1, used_backends=["bm25s"])[0][
        0
    ].context_id == "doc-beta"
    assert retriever.search("alpha", top_k=1, used_backends=["faiss"])[0][
        0
    ].source == "memory"
    retriever.close()

    reopened = FlexRetriever.from_backends(
        {
            "bm25s": BM25SBackend(None, tmp_path / "bm25s"),
            "faiss": FaissBackend(
                None,
                tmp_path / "faiss",
                query_encoder=TokenEncoder(),
            ),
        },
        context_store=LMDBContextStore(LMDBContextStoreConfig(path=tmp_path / "store")),
    )
    assert reopened.backends["bm25s"].view == text_view
    assert reopened.backends["faiss"].view == dense_view
    assert reopened.search("alpha", top_k=1, used_backends=["faiss"])[0][
        0
    ].context_id == "doc-alpha"

    reopened.add_contexts(
        [Context(context_id="doc-delta", data={"title": "delta", "text": "delta"})]
    )
    assert reopened.search("delta", top_k=1, used_backends=["bm25s"])[0][
        0
    ].context_id == "doc-delta"
    assert reopened.search("delta", top_k=1, used_backends=["faiss"])[0][
        0
    ].context_id == "doc-delta"
    reopened.clear()
    assert reopened.count() == 0
    reopened.close()
