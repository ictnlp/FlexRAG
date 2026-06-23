import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from flexrag.common import ChatTurn
from flexrag.common.dataclasses import RetrievedContext
from flexrag.models.tokenizer import SpaceTokenizerConfig
from flexrag.processors.rankers import (
    HFRankerConfig,
    LiteLLMRankerConfig,
    RankGPTRankerConfig,
)
from flexrag.processors.refiners import (
    AbstractiveSummarizerConfig,
    ContextArrangerConfig,
    RecompExtractiveSummarizerConfig,
)
from flexrag.resources import (
    EncoderHandle,
    GeneratorHandle,
    RankerHandle,
    RefinerHandle,
    ResourceManager,
    ResourceManagerConfig,
    Resources,
    ResourceSpec,
    ScorerHandle,
    TokenizerHandle,
)
from flexrag.resources.runtime_adapters import (
    DirectRuntimeAdapter,
    ProcessRuntimeAdapter,
)
from flexrag.retrievers.index import RetrieverIndexConfig


@dataclass
class FakeEncoderConfig:
    name: str
    embedding_size: int = 8


@dataclass
class FakeGeneratorConfig:
    name: str


@dataclass
class FakeScorerConfig:
    name: str


@dataclass
class FakeTokenizerConfig:
    name: str
    vocab_size: int = 1024


@dataclass
class UnknownConfig:
    name: str = "unknown"


@dataclass
class UnsupportedConfig:
    name: str = "unsupported"


class FakeEncoderImpl:
    pass


class FakeGeneratorImpl:
    pass


class FakeScorerImpl:
    pass


class FakeTokenizerImpl:
    def __init__(
        self,
        config: FakeTokenizerConfig,
        *,
        close_log: list[str] | None = None,
    ) -> None:
        self.config = config
        self.close_log = close_log
        return

    def tokenize(self, text: str) -> list[str]:
        return text.split()

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def encode(self, text: str) -> list[int]:
        return [len(token) for token in self.tokenize(text)]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(f"T{token}" for token in tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [len(token) for token in tokens]

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        return [f"T{token_id}" for token_id in token_ids]

    @property
    def reversible(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    def close(self) -> None:
        if self.close_log is not None:
            self.close_log.append(f"tokenizer:{self.config.name}")
        return


class UnsupportedImpl:
    pass


class _FakeProcessRuntime(ProcessRuntimeAdapter):
    def run_sync(self, coro):
        return asyncio.run(coro)

    async def run_async(self, coro):
        return await coro


class FakeEncoderRuntime(_FakeProcessRuntime):
    def __init__(
        self,
        config: FakeEncoderConfig,
        impl_cls: type[FakeEncoderImpl] | None = None,
        *,
        constructor_log: list[dict[str, Any]] | None = None,
        close_log: list[str] | None = None,
        tag: str | None = None,
    ) -> None:
        self.config = config
        self.impl_cls = impl_cls
        self.close_log = close_log
        self.tag = tag
        if constructor_log is not None:
            constructor_log.append(
                {
                    "name": config.name,
                    "impl_cls": impl_cls,
                    "tag": tag,
                }
            )
        return

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> np.ndarray:
        if method != "_encode_batch":
            raise AttributeError(method)
        batch = args[0]
        return np.full((len(batch), self.config.embedding_size), fill_value=len(batch))

    async def agetattr(self, name: str) -> Any:
        if name == "embedding_size":
            return self.config.embedding_size
        raise AttributeError(name)

    def close(self) -> None:
        if self.close_log is not None:
            self.close_log.append(self.config.name)
        return


class FakeGeneratorRuntime(_FakeProcessRuntime):
    def __init__(
        self,
        config: FakeGeneratorConfig,
        impl_cls: type[FakeGeneratorImpl] | None = None,
        *,
        constructor_log: list[dict[str, Any]] | None = None,
        close_log: list[str] | None = None,
    ) -> None:
        self.config = config
        self.impl_cls = impl_cls
        self.close_log = close_log
        if constructor_log is not None:
            constructor_log.append(
                {
                    "name": config.name,
                    "impl_cls": impl_cls,
                }
            )
        return

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> Any:
        if method == "_generate_batch":
            prefixes = args[0]
            return [[f"{self.config.name}:{prefix}"] for prefix in prefixes]
        if method == "_chat_batch":
            messages = args[0]
            return [[ChatTurn(role="assistant", content="1 2")] for _ in messages]
        raise AttributeError(method)

    def close(self) -> None:
        if self.close_log is not None:
            self.close_log.append(self.config.name)
        return


class FakeScorerRuntime(_FakeProcessRuntime):
    def __init__(
        self,
        config: FakeScorerConfig,
        impl_cls: type[FakeScorerImpl] | None = None,
        *,
        close_log: list[str] | None = None,
    ) -> None:
        self.config = config
        self.impl_cls = impl_cls
        self.close_log = close_log
        return

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> np.ndarray:
        if method != "_score_batch":
            raise AttributeError(method)
        pairs = args[0]
        return np.arange(len(pairs), dtype=float)

    def close(self) -> None:
        if self.close_log is not None:
            self.close_log.append(f"close:{self.config.name}")
        return

    async def aclose(self) -> None:
        if self.close_log is not None:
            self.close_log.append(f"aclose:{self.config.name}")
        return


Resources.register(
    "test_resource_manager_fake_encoder",
    interface="encoder",
    config_class=FakeEncoderConfig,
    runtime_adapter_cls=FakeEncoderRuntime,
)(FakeEncoderImpl)

Resources.register(
    "test_resource_manager_fake_generator",
    interface="generator",
    config_class=FakeGeneratorConfig,
    runtime_adapter_cls=FakeGeneratorRuntime,
)(FakeGeneratorImpl)

Resources.register(
    "test_resource_manager_fake_scorer",
    interface="scorer",
    config_class=FakeScorerConfig,
    runtime_adapter_cls=FakeScorerRuntime,
)(FakeScorerImpl)

Resources.register(
    "test_resource_manager_fake_tokenizer",
    interface="tokenizer",
    config_class=FakeTokenizerConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(FakeTokenizerImpl)

Resources.register(
    "test_resource_manager_unsupported",
    interface="unsupported",
    config_class=UnsupportedConfig,
    runtime_adapter_cls=DirectRuntimeAdapter,
)(UnsupportedImpl)


def _encoder_spec(
    name: str,
    *,
    runtime_kwargs: dict[str, Any] | None = None,
) -> ResourceSpec:
    return ResourceSpec(
        name=name,
        config=FakeEncoderConfig(name=name),
        runtime_kwargs=runtime_kwargs or {},
    )


def test_resource_manager_lazy_load_and_reuse():
    constructor_log: list[dict[str, Any]] = []
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                _encoder_spec(
                    "first",
                    runtime_kwargs={"constructor_log": constructor_log},
                ),
                _encoder_spec(
                    "second",
                    runtime_kwargs={"constructor_log": constructor_log},
                ),
            ]
        )
    )

    first = resources.get("first")

    assert first is resources.get("first")
    assert isinstance(first, EncoderHandle)
    assert constructor_log == [
        {"name": "first", "impl_cls": FakeEncoderImpl, "tag": None}
    ]


def test_resource_manager_preload_and_close_order():
    close_log: list[str] = []
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                _encoder_spec("first", runtime_kwargs={"close_log": close_log}),
                _encoder_spec("second", runtime_kwargs={"close_log": close_log}),
            ],
            preload=["first"],
        )
    )

    resources.get("second")
    resources.close()
    resources.close()

    assert close_log == ["second", "first"]


@pytest.mark.asyncio
async def test_resource_manager_aclose_order_uses_async_close():
    close_log: list[str] = []
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(
                    name="first",
                    config=FakeScorerConfig(name="first"),
                    runtime_kwargs={"close_log": close_log},
                ),
                ResourceSpec(
                    name="second",
                    config=FakeScorerConfig(name="second"),
                    runtime_kwargs={"close_log": close_log},
                ),
            ]
        )
    )

    resources.get("first")
    resources.get("second")
    await resources.aclose()
    await resources.aclose()

    assert close_log == ["aclose:second", "aclose:first"]


def test_resource_manager_encoder_handle_calls_sync_and_async():
    resources = ResourceManager.load(
        ResourceManagerConfig(resources=[_encoder_spec("dense")])
    )

    encoder = resources.get("dense")

    assert encoder.embedding_size == 8
    assert encoder.encode(["a", "b"]).shape == (2, 8)
    assert asyncio.run(encoder.async_encode(["a"])).shape == (1, 8)
    assert not hasattr(encoder, "close")
    assert not hasattr(encoder, "aclose")
    assert not hasattr(encoder, "_close")
    assert not hasattr(encoder, "_aclose")


def test_resource_manager_passes_runtime_kwargs_to_adapter():
    constructor_log: list[dict[str, Any]] = []
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                _encoder_spec(
                    "dense",
                    runtime_kwargs={
                        "constructor_log": constructor_log,
                        "tag": "runtime-policy",
                    },
                )
            ]
        )
    )

    resources.get("dense")

    assert constructor_log == [
        {"name": "dense", "impl_cls": FakeEncoderImpl, "tag": "runtime-policy"}
    ]


def test_resource_manager_maps_generator_and_scorer_handles():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="generator", config=FakeGeneratorConfig("generator")),
                ResourceSpec(name="scorer", config=FakeScorerConfig("scorer")),
            ]
        )
    )

    generator = resources.get("generator")
    scorer = resources.get("scorer")

    assert isinstance(generator, GeneratorHandle)
    assert isinstance(scorer, ScorerHandle)
    assert generator.generate("hello") == [["generator:hello"]]
    assert scorer.score(("q", "d")).shape == (1,)


def test_resource_manager_maps_tokenizer_handle():
    close_log: list[str] = []
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(
                    name="tokenizer",
                    config=FakeTokenizerConfig("fake", vocab_size=42),
                    runtime_kwargs={"close_log": close_log},
                )
            ]
        )
    )

    tokenizer = resources.get("tokenizer")

    assert isinstance(tokenizer, TokenizerHandle)
    assert tokenizer.tokenize("hello world") == ["hello", "world"]
    assert tokenizer.detokenize(["hello", "world"]) == "hello world"
    assert tokenizer.encode("hello world") == [5, 5]
    assert tokenizer.decode([1, 2]) == "T1 T2"
    assert tokenizer.tokens_to_ids(["hi", "there"]) == [2, 5]
    assert tokenizer.ids_to_tokens([2, 5]) == ["T2", "T5"]
    assert tokenizer.reversible is True
    assert tokenizer.vocab_size == 42

    resources.close()
    assert close_log == ["tokenizer:fake"]


def test_resource_manager_manages_builtin_space_tokenizer():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="tokenizer", config=SpaceTokenizerConfig()),
            ]
        )
    )

    tokenizer = resources.get("tokenizer")

    assert isinstance(tokenizer, TokenizerHandle)
    assert tokenizer.tokenize("a b") == ["a", "b"]
    assert tokenizer.detokenize(["a", "b"]) == "a b"
    assert tokenizer.reversible is False
    assert tokenizer.vocab_size == 0


def test_resource_manager_constructs_hf_ranker_with_scorer_ref():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="scorer", config=FakeScorerConfig("scorer")),
                ResourceSpec(
                    name="ranker",
                    config=HFRankerConfig(),
                    refs={"scorer": "scorer"},
                ),
            ]
        )
    )

    ranker = resources.get("ranker")
    result = ranker.rank("query", ["first", "second", "third"])
    async_result = asyncio.run(
        ranker.async_rank("query", ["first", "second", "third"])
    )

    assert isinstance(ranker, RankerHandle)
    assert result.candidates == ["third", "second", "first"]
    assert async_result.candidates == result.candidates


def test_resource_manager_constructs_rank_gpt_ranker_with_generator_ref():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="generator", config=FakeGeneratorConfig("generator")),
                ResourceSpec(
                    name="ranker",
                    config=RankGPTRankerConfig(window_size=2, step_size=1),
                    refs={"generator": "generator"},
                ),
            ]
        )
    )

    ranker = resources.get("ranker")
    result = ranker.rank("query", ["first", "second"])

    assert isinstance(ranker, RankerHandle)
    assert result.candidates == ["first", "second"]


@pytest.mark.asyncio
async def test_resource_manager_constructs_litellm_ranker_with_remote_runtime(
    monkeypatch,
):
    import litellm

    active = 0
    max_seen = 0

    async def fake_arerank(
        *,
        model,
        query,
        documents,
        top_n,
        return_documents,
        **request_kwargs,
    ):
        nonlocal active, max_seen
        assert model == "cohere/rerank-v3.5"
        assert top_n == len(documents)
        assert not return_documents
        assert request_kwargs["api_key"] == "test"
        active += 1
        max_seen = max(max_seen, active)
        await asyncio.sleep(0.01)
        active -= 1
        return SimpleNamespace(
            results=[
                SimpleNamespace(index=0, relevance_score=0.1),
                SimpleNamespace(index=1, relevance_score=0.9),
            ]
        )

    monkeypatch.setattr(litellm, "arerank", fake_arerank)
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(
                    name="ranker",
                    config=LiteLLMRankerConfig(
                        provider="cohere",
                        model_name="rerank-v3.5",
                        api_key="test",
                    ),
                    runtime_kwargs={"max_concurrency": 2},
                )
            ]
        )
    )

    try:
        ranker = resources.get("ranker")
        results = await asyncio.gather(
            *[
                ranker.async_rank(f"query-{idx}", ["first", "second"])
                for idx in range(4)
            ]
        )
    finally:
        resources.close()

    assert isinstance(ranker, RankerHandle)
    assert all(result.candidates == ["second", "first"] for result in results)
    assert max_seen <= 2


def test_resource_manager_constructs_context_arranger_refiner():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(
                    name="refiner",
                    config=ContextArrangerConfig(order="descending"),
                ),
            ]
        )
    )

    refiner = resources.get("refiner")
    contexts = [
        RetrievedContext(context_id="low", data={"text": "low"}, score=0.1),
        RetrievedContext(context_id="high", data={"text": "high"}, score=0.9),
    ]
    refined = refiner.refine(contexts)

    assert isinstance(refiner, RefinerHandle)
    assert [ctx.context_id for ctx in refined] == ["high", "low"]


def test_resource_manager_constructs_abstractive_summarizer_with_generator_ref():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="generator", config=FakeGeneratorConfig("summary")),
                ResourceSpec(
                    name="refiner",
                    config=AbstractiveSummarizerConfig(refined_field="text"),
                    refs={"generator": "generator"},
                ),
            ]
        )
    )

    refiner = resources.get("refiner")
    contexts = [
        RetrievedContext(
            context_id="ctx",
            query="query",
            data={"text": "original text"},
            score=1.0,
        )
    ]
    refined = refiner.refine(contexts)

    assert isinstance(refiner, RefinerHandle)
    assert refined[0].data["text"] == "summary:original text"
    assert contexts[0].data["text"] == "original text"


def test_resource_manager_constructs_extractive_summarizer_with_encoder_ref():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="encoder", config=FakeEncoderConfig("encoder")),
                ResourceSpec(
                    name="refiner",
                    config=RecompExtractiveSummarizerConfig(
                        preserved_sents=1,
                        refined_field="text",
                    ),
                    refs={"encoder": "encoder"},
                ),
            ]
        )
    )

    refiner = resources.get("refiner")
    contexts = [
        RetrievedContext(
            context_id="ctx",
            query="query",
            data={"text": "First sentence is useful. Second sentence is extra."},
            score=1.0,
        )
    ]
    refined = refiner.refine(contexts)

    assert isinstance(refiner, RefinerHandle)
    assert refined[0].data["text_summary"]
    assert contexts[0].data.get("text_summary") is None


def test_resource_manager_injects_refs_as_handles():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="scorer", config=FakeScorerConfig("scorer")),
                ResourceSpec(
                    name="ranker",
                    config=HFRankerConfig(),
                    refs={"scorer": "scorer"},
                ),
            ]
        )
    )

    ranker = resources.get("ranker")
    scorer = resources.get("scorer")

    assert resources._resources["ranker"]._resource.scorer is scorer
    assert ranker.rank("query", ["first", "second"]).candidates == [
        "second",
        "first",
    ]


def test_resource_manager_rejects_ref_and_runtime_kwarg_conflict():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(name="scorer", config=FakeScorerConfig("scorer")),
                ResourceSpec(
                    name="ranker",
                    config=HFRankerConfig(),
                    refs={"scorer": "scorer"},
                    runtime_kwargs={"scorer": object()},
                ),
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate constructor kwargs"):
        resources.get("ranker")


def test_resource_manager_rejects_refs_for_non_direct_runtime():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                _encoder_spec("encoder"),
                ResourceSpec(
                    name="generator",
                    config=FakeGeneratorConfig("generator"),
                    refs={"encoder": "encoder"},
                ),
            ]
        )
    )

    with pytest.raises(ValueError, match="only direct runtime"):
        resources.get("generator")


def test_resource_manager_rejects_missing_resource_and_ref():
    resources = ResourceManager.load(ResourceManagerConfig())

    with pytest.raises(KeyError, match="Resource not found"):
        resources.get("missing")

    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[
                ResourceSpec(
                    name="ranker",
                    config=HFRankerConfig(),
                    refs={"scorer": "missing"},
                )
            ]
        )
    )
    with pytest.raises(KeyError, match="Resource not found"):
        resources.get("ranker")


def test_resource_manager_rejects_duplicate_resource_names():
    with pytest.raises(ValueError, match="Duplicate resource name"):
        ResourceManager.load(
            ResourceManagerConfig(
                resources=[_encoder_spec("dense"), _encoder_spec("dense")]
            )
        )


def test_resource_spec_fills_resource_for_concrete_config():
    spec = ResourceSpec(name="dense", config=FakeEncoderConfig(name="dense"))

    assert spec.resource == "test_resource_manager_fake_encoder"
    assert isinstance(spec.config, FakeEncoderConfig)


def test_resource_manager_config_loads_resource_spec_from_yaml():
    cfg = ResourceManagerConfig.loads(
        """
resources:
  - name: dense
    resource: test_resource_manager_fake_encoder
    config:
      name: dense
      embedding_size: 16
    runtime_kwargs:
      tag: yaml-runtime
"""
    )

    assert len(cfg.resources) == 1
    spec = cfg.resources[0]
    assert spec.name == "dense"
    assert spec.resource == "test_resource_manager_fake_encoder"
    assert spec.config == FakeEncoderConfig(name="dense", embedding_size=16)

    resources = ResourceManager.load(cfg)
    encoder = resources.get("dense")
    assert encoder.embedding_size == 16


def test_resource_manager_config_dumps_resource_discriminator_for_round_trip():
    cfg = ResourceManagerConfig(
        resources=[
            ResourceSpec(
                name="dense",
                config=FakeEncoderConfig(name="dense", embedding_size=16),
            )
        ]
    )

    dumped = cfg.dumps()
    loaded = ResourceManagerConfig.loads(dumped)

    assert "test_resource_manager_fake_encoder" in dumped
    assert loaded.resources[0].resource == "test_resource_manager_fake_encoder"
    assert loaded.resources[0].config == FakeEncoderConfig(
        name="dense",
        embedding_size=16,
    )


def test_resource_spec_rejects_dict_config_without_resource():
    with pytest.raises(ValueError, match="resource is required"):
        ResourceSpec(name="dense", config={"name": "dense"})


def test_resource_spec_rejects_mismatched_resource_and_config():
    with pytest.raises(ValueError, match="does not match config class"):
        ResourceSpec(
            name="dense",
            resource="test_resource_manager_fake_generator",
            config=FakeEncoderConfig(name="dense"),
        )


def test_resource_spec_rejects_unregistered_config_and_index_config():
    with pytest.raises(KeyError, match="not registered"):
        ResourceSpec(name="unknown", config=UnknownConfig())

    with pytest.raises(KeyError, match="not registered"):
        ResourceSpec(name="index", config=RetrieverIndexConfig())


def test_resource_manager_rejects_interface_without_handle_mapping():
    resources = ResourceManager.load(
        ResourceManagerConfig(
            resources=[ResourceSpec(name="bad", config=UnsupportedConfig())]
        )
    )

    with pytest.raises(TypeError, match="Resource interface is not supported"):
        resources.get("bad")
