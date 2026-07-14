from __future__ import annotations

from pathlib import Path

import pytest

from flexrag.common.dataclasses import ChatMessages, ChatTurn, Context, RetrievedContext
from flexrag.models.generators import GenerationConfig
from flexrag.resources import (
    ResourceManager,
    Resources,
    ResourcesConfig,
    ResourceSpec,
    _ResourceRegister,
)
from flexrag.resources.handles import HANDLE_TYPES
from flexrag.retrievers.view import RetrievalViewConfig
from tests.resources.support.registry import FAKE_RESOURCES


def contexts() -> list[Context]:
    return [
        Context(context_id="doc-a", data={"text": "alpha beta"}),
        Context(context_id="doc-b", data={"text": "beta gamma"}),
        Context(context_id="doc-c", data={"text": "delta epsilon"}),
    ]


def view_config() -> RetrievalViewConfig:
    return RetrievalViewConfig(name="text", fields=["text"])


def merged_registry() -> _ResourceRegister:
    registry = _ResourceRegister()
    for entry in (*Resources.entries, *FAKE_RESOURCES.entries):
        registry.register(
            entry.resource_name,
            interface=entry.interface,
            config_class=entry.config_class,
            default_runtime=entry.default_runtime,
            parallel_safe=entry.parallel_safe,
            batching=entry.batching,
        )(entry.raw_cls)
    return registry


def test_formal_resource_entries_are_registered() -> None:
    expected = {
        "litellm_encoder": ("encoder", "async"),
        "sentence_transformer_encoder": ("encoder", "process"),
        "hf_encoder": ("encoder", "process"),
        "hf_clip_encoder": ("encoder", "process"),
        "litellm_generator": ("generator", "async"),
        "hf_generator": ("generator", "process"),
        "hf_cross_encoder_scorer": ("scorer", "process"),
        "hf_logits_scorer": ("scorer", "process"),
        "hf_colbert_scorer": ("scorer", "process"),
        "hf_ranker": ("ranker", "async"),
        "rank_gpt_ranker": ("ranker", "direct"),
        "litellm_ranker": ("ranker", "async"),
        "lmdb_context_store": ("context_store", "direct"),
        "sqlite_context_store": ("context_store", "direct"),
        "bm25s_backend": ("collection_backend", "process"),
        "faiss_backend": ("collection_backend", "process"),
        "elastic_backend": ("collection_backend", "async"),
        "lance_backend": ("collection_backend", "async"),
        "flex_retriever": ("retriever", "direct"),
        "space_tokenizer": ("tokenizer", "direct"),
        "moses_tokenizer": ("tokenizer", "direct"),
        "nltk_tokenizer": ("tokenizer", "direct"),
        "jieba_tokenizer": ("tokenizer", "direct"),
        "hf_tokenizer": ("tokenizer", "direct"),
        "tiktoken_tokenizer": ("tokenizer", "direct"),
        "char_chunker": ("chunker", "direct"),
        "token_chunker": ("chunker", "direct"),
        "recursive_chunker": ("chunker", "direct"),
        "regex_sentence_splitter": ("chunker", "direct"),
        "nltk_sentence_splitter": ("chunker", "direct"),
        "spacy_sentence_splitter": ("chunker", "direct"),
        "sentence_chunker": ("chunker", "direct"),
        "semantic_chunker": ("chunker", "direct"),
        "lumber_chunker": ("chunker", "direct"),
        "densex_chunker": ("chunker", "direct"),
        "context_arranger": ("refiner", "direct"),
        "abstractive_summarizer": ("refiner", "direct"),
        "extractive_summarizer": ("refiner", "direct"),
    }
    for resource_name, (interface, runtime) in expected.items():
        entry = Resources.resolve_name(resource_name)
        assert entry.interface == interface
        assert entry.default_runtime == runtime
        assert entry.interface in HANDLE_TYPES


def test_litellm_encoder_and_generator_smoke(mock_litellm_client) -> None:
    resources = ResourceManager(
        [
            ResourceSpec(
                name="encoder",
                resource_name="litellm_encoder",
                resource_config={
                    "provider": "openai",
                    "model_name": "text-embedding-3-small",
                    "embedding_size": 4,
                },
            ),
            ResourceSpec(
                name="generator",
                resource_name="litellm_generator",
                resource_config={
                    "provider": "openai",
                    "model_name": "gpt-4o-mini",
                },
            ),
        ]
    )
    try:
        encoder = resources.get("encoder")
        generator = resources.get("generator")

        assert encoder.encode(["alpha", "beta"]).shape == (2, 4)
        assert encoder.embedding_size == 4
        assert generator.generate("prompt", GenerationConfig(do_sample=False)) == [
            ["Mocked LiteLLM text completion 0"]
        ]
        messages = [ChatMessages(history=[ChatTurn(role="user", content="Ping")])]
        assert generator.chat(messages)[0][0].text_content == (
            "Mocked LiteLLM chat response 0"
        )
        assert mock_litellm_client["calls"]["aembedding"][0]["model"] == (
            "openai/text-embedding-3-small"
        )
    finally:
        resources.close()


@pytest.mark.asyncio
async def test_sqlite_context_store_persists_and_bridges_async(tmp_path: Path) -> None:
    spec = ResourceSpec(
        name="store",
        resource_name="sqlite_context_store",
        resource_config={"path": tmp_path / "store.db"},
    )
    resources = ResourceManager([spec])
    try:
        store = resources.get("store")
        await store.async_set_many(contexts())
        assert await store.async_count() == 3
        assert (await store.async_get("doc-b")).data["text"] == "beta gamma"
        assert [ctx.context_id async for ctx in store.async_iter_contexts()] == [
            "doc-a",
            "doc-b",
            "doc-c",
        ]
        snapshot = store.iter_contexts()
        store.clear()
        assert [ctx.context_id for ctx in snapshot] == [
            "doc-a",
            "doc-b",
            "doc-c",
        ]
        store.set_many(contexts())
        assert [ctx.context_id for ctx in await store.async_get_all()] == [
            "doc-a",
            "doc-b",
            "doc-c",
        ]
    finally:
        await resources.async_close()

    reopened = ResourceManager([spec])
    try:
        store = reopened.get("store")
        assert store.count() == 3
        store.clear()
        assert store.count() == 0
    finally:
        reopened.close()


def test_bm25s_and_faiss_backend_smoke(tmp_path: Path, mock_litellm_client) -> None:
    resources = ResourceManager(
        [
            ResourceSpec(
                name="bm25",
                resource_name="bm25s_backend",
                resource_config={
                    "path": tmp_path / "bm25",
                    "view": view_config(),
                },
            ),
            ResourceSpec(
                name="encoder",
                resource_name="litellm_encoder",
                resource_config={
                    "provider": "openai",
                    "model_name": "text-embedding-3-small",
                    "embedding_size": 4,
                },
            ),
            ResourceSpec(
                name="faiss",
                resource_name="faiss_backend",
                resource_config={
                    "path": tmp_path / "faiss",
                    "view": view_config(),
                    "distance_function": "IP",
                },
                refs={"query_encoder": "encoder"},
            ),
        ]
    )
    try:
        bm25 = resources.get("bm25")
        bm25.rebuild(contexts())
        assert bm25.search_hits(["alpha"], top_k=1)[0][0].context_id == "doc-a"
        assert bm25.count() == 3
        bm25.clear()
        assert bm25.count() == 0

        faiss = resources.get("faiss")
        faiss.rebuild(contexts())
        assert faiss.search_hits(["alpha"], top_k=1)[0]
        assert mock_litellm_client["calls"]["aembedding"]
    finally:
        resources.close()


def test_flex_retriever_resource_smoke(tmp_path: Path) -> None:
    resources = ResourceManager.load(
        ResourcesConfig(
            resources=[
                ResourceSpec(
                    name="store",
                    resource_name="sqlite_context_store",
                    resource_config={"path": tmp_path / "store.db"},
                ),
                ResourceSpec(
                    name="bm25",
                    resource_name="bm25s_backend",
                    resource_config={
                        "path": tmp_path / "bm25",
                        "view": view_config(),
                    },
                ),
                ResourceSpec(
                    name="retriever",
                    resource_name="flex_retriever",
                    refs={
                        "backends": {"sparse": "bm25"},
                        "context_store": "store",
                    },
                ),
            ],
            preload=["retriever"],
        )
    )
    try:
        assert list(resources._handles) == ["bm25", "store", "retriever"]
        retriever = resources.get("retriever")
        retriever.add_contexts(iter(contexts()))
        assert retriever.count() == 3
        assert retriever.list_backends() == ["sparse"]
        assert retriever.search("alpha", top_k=1)[0][0].context_id == "doc-a"
    finally:
        resources.close()


@pytest.mark.asyncio
async def test_ranker_tokenizer_chunker_and_refiner_smoke() -> None:
    registry = merged_registry()
    resources = ResourceManager(
        [
            ResourceSpec(
                name="generator",
                resource_name="fake_generator",
                resource_config={"chat_response": "2 1"},
            ),
            ResourceSpec(
                name="ranker",
                resource_name="rank_gpt_ranker",
                resource_config={"window_size": 2, "step_size": 1},
                refs={"generator": "generator"},
            ),
            ResourceSpec(name="tokenizer", resource_name="space_tokenizer"),
            ResourceSpec(
                name="chunker",
                resource_name="regex_sentence_splitter",
            ),
            ResourceSpec(
                name="refiner",
                resource_name="context_arranger",
                resource_config={"order": "descending"},
            ),
        ],
        registry=registry,
    )
    try:
        assert resources.get("ranker").rank(
            "query", ["first", "second"]
        ).candidates == [
            "second",
            "first",
        ]
        assert resources.get("tokenizer").tokenize("alpha beta") == ["alpha", "beta"]
        assert [
            chunk.text for chunk in resources.get("chunker").chunk("First. Second.")
        ] == [
            "First.",
            "Second.",
        ]
        contexts_to_refine = [
            RetrievedContext(
                context_id="low", query="q", data={"text": "a"}, score=1.0
            ),
            RetrievedContext(
                context_id="high", query="q", data={"text": "b"}, score=2.0
            ),
        ]
        refiner = resources.get("refiner")
        refined = refiner.refine(contexts_to_refine)
        async_refined = await refiner.async_refine(contexts_to_refine)
        assert [context.context_id for context in refined] == ["high", "low"]
        assert async_refined == refined
    finally:
        resources.close()
