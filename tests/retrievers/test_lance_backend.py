from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb")

from flexrag.common import Context
from flexrag.retrievers import (
    FlexRetriever,
    LanceBackend,
    LanceBackendConfig,
    RetrievalViewConfig,
)


class TokenEncoder:
    def encode(self, items: list[Any]) -> np.ndarray:
        return np.asarray(
            [
                [float("alpha" in str(item).lower()), float("beta" in str(item).lower())]
                for item in items
            ],
            dtype="float32",
        )


def test_lance_backend_dense_native_round_trip(tmp_path: Path) -> None:
    lazy_backend = LanceBackend(
        LanceBackendConfig(
            uri=tmp_path / "lazy",
            view=RetrievalViewConfig(name="dense", fields=["text"]),
            table_name="contexts",
            retrieval_mode="dense",
        ),
        query_encoder=TokenEncoder(),
    )
    assert lazy_backend.view is not None
    assert lazy_backend.client is None
    lazy_backend.close()

    backend = LanceBackend(
        LanceBackendConfig(
            uri=tmp_path / "db",
            view=RetrievalViewConfig(name="dense", fields=["text"]),
            table_name="contexts",
            retrieval_mode="dense",
        ),
        query_encoder=TokenEncoder(),
    )
    assert backend.view is not None
    retriever = FlexRetriever.from_backends({"lance": backend})
    retriever.add_contexts(
        [
            Context(context_id="doc-alpha", data={"text": "alpha vector"}),
            Context(context_id="doc-beta", data={"text": "beta vector"}),
        ]
    )
    assert retriever.count() == 2
    assert retriever.search("alpha", top_k=1)[0][0].context_id == "doc-alpha"
    backend.close()

    restored = LanceBackend(
        LanceBackendConfig(
            uri=tmp_path / "db",
            table_name="contexts",
            retrieval_mode="dense",
        ),
        query_encoder=TokenEncoder(),
    )
    assert restored.view is None
    restored_retriever = FlexRetriever.from_backends({"lance": restored})
    assert restored_retriever.search("alpha", top_k=1)[0][0].context_id == "doc-alpha"
    assert restored.view is not None
    restored.close()
