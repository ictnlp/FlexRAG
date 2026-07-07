from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from flexrag.common import Context
from flexrag.retrievers import (
    ElasticBackend,
    ElasticBackendConfig,
    FlexRetriever,
    LMDBContextStore,
    LMDBContextStoreConfig,
    RetrievalView,
    RetrievalViewConfig,
)
from flexrag.retrievers.backends.elastic import (
    INTERNAL_PAYLOAD,
    INTERNAL_TEXT,
    INTERNAL_VECTOR,
)
from tests.support.fixtures.elastic import FakeElasticClient

TEXT_VIEW = RetrievalView("text", ["title", "body"])
DENSE_VIEW = RetrievalView("dense", ["tokens"])
TEXT_VIEW_CONFIG = RetrievalViewConfig(name="text", fields=["title", "body"])
DENSE_VIEW_CONFIG = RetrievalViewConfig(name="dense", fields=["tokens"])
TOKENS = ("alpha", "beta", "gamma")


class TokenEncoder:
    def encode(self, items: list[Any]) -> np.ndarray:
        return np.asarray(
            [[float(token in _text(item)) for token in TOKENS] for item in items],
            dtype="float32",
        )


def contexts() -> list[Context]:
    return [_context("alpha"), _context("beta")]


def _context(label: str) -> Context:
    return Context(
        context_id=f"doc-{label}",
        data={
            "title": label,
            "body": f"{label} sparse text",
            "tokens": [label, "dense"],
        },
    )


def _text(item: Any) -> str:
    return " ".join(map(str, item)).lower() if isinstance(item, list) else str(item)


def user_docs(client: FakeElasticClient, index: str) -> list[dict[str, Any]]:
    return [
        doc
        for doc in client.docs.get(index, {}).values()
        if doc.get("_context_id") != "__flexrag_backend_meta__"
    ]


def test_sparse_native_payload(fake_elastic_client: FakeElasticClient) -> None:
    config = ElasticBackendConfig(
        index_name="sparse",
        view=TEXT_VIEW_CONFIG,
        retrieval_mode="sparse",
    )
    retriever = FlexRetriever.from_backends(
        {"es": ElasticBackend(config, client=fake_elastic_client)}
    )
    retriever.add_contexts(contexts())
    assert retriever.count() == 2
    assert retriever.search("beta", top_k=1)[0][0].context_id == "doc-beta"
    assert all(
        {INTERNAL_TEXT, INTERNAL_PAYLOAD} <= doc.keys()
        for doc in user_docs(fake_elastic_client, "sparse")
    )

    restored = ElasticBackend(
        ElasticBackendConfig(index_name="sparse", retrieval_mode="sparse"),
        client=fake_elastic_client,
    )
    assert restored.view == TEXT_VIEW
    with pytest.raises(ValueError):
        ElasticBackend(
            ElasticBackendConfig(
                index_name="sparse",
                view=RetrievalViewConfig(name="other", fields=["body"]),
                retrieval_mode="sparse",
            ),
            client=fake_elastic_client,
        )


def test_sparse_external_context_store(
    tmp_path: Path,
    fake_elastic_client: FakeElasticClient,
) -> None:
    store = LMDBContextStore(LMDBContextStoreConfig(path=tmp_path / "elastic-store"))
    config = ElasticBackendConfig(
        index_name="external",
        view=TEXT_VIEW_CONFIG,
        retrieval_mode="sparse",
        store_payload=False,
    )
    retriever = FlexRetriever.from_backends(
        {"es": ElasticBackend(config, client=fake_elastic_client)},
        context_store=store,
    )
    retriever.add_contexts(contexts())
    assert retriever.search("alpha", top_k=1)[0][0].data["title"] == "alpha"
    assert all(
        INTERNAL_PAYLOAD not in doc for doc in user_docs(fake_elastic_client, "external")
    )
    store.close()


def test_dense_native_payload_and_schema(
    fake_elastic_client: FakeElasticClient,
) -> None:
    config = ElasticBackendConfig(
        index_name="dense",
        view=DENSE_VIEW_CONFIG,
        retrieval_mode="dense",
        index_options={"type": "int8_hnsw"},
        search_options={"num_candidates": 5},
    )
    backend = ElasticBackend(
        config,
        client=fake_elastic_client,
        query_encoder=TokenEncoder(),
    )
    retriever = FlexRetriever.from_backends({"dense": backend})
    retriever.add_contexts(contexts())
    assert retriever.search(["alpha"], top_k=1)[0][0].context_id == "doc-alpha"
    mapping = fake_elastic_client.created_mappings["dense"]["mappings"]["properties"]
    assert mapping[INTERNAL_VECTOR]["dims"] == 3
    assert mapping[INTERNAL_VECTOR]["index_options"] == {"type": "int8_hnsw"}
    assert all(
        {INTERNAL_VECTOR, INTERNAL_PAYLOAD} <= doc.keys()
        for doc in user_docs(fake_elastic_client, "dense")
    )
    knn = fake_elastic_client.msearch_bodies[-1][1]["knn"]
    assert knn["num_candidates"] == 5
    assert knn["filter"]

    restored = ElasticBackend(
        ElasticBackendConfig(
            index_name="dense",
            retrieval_mode="dense",
            index_options={"type": "int8_hnsw"},
            search_options={"num_candidates": 5},
        ),
        client=fake_elastic_client,
        query_encoder=TokenEncoder(),
    )
    assert restored.view == DENSE_VIEW
    with pytest.raises(ValueError):
        ElasticBackend(
            ElasticBackendConfig(index_name="dense", retrieval_mode="sparse"),
            client=fake_elastic_client,
        )
