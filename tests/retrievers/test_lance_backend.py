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
    backend = LanceBackend(
        LanceBackendConfig(
            uri=tmp_path / "db",
            view=RetrievalViewConfig(name="dense", fields=["text"]),
            table_name="contexts",
            retrieval_mode="dense",
        ),
        query_encoder=TokenEncoder(),
    )
    retriever = FlexRetriever.from_backends({"lance": backend})
    retriever.add_contexts(
        [
            Context(context_id="doc-alpha", data={"text": "alpha vector"}),
            Context(context_id="doc-beta", data={"text": "beta vector"}),
        ]
    )
    assert retriever.count() == 2
    assert retriever.search("alpha", top_k=1)[0][0].context_id == "doc-alpha"
    retriever.close()
