import numpy as np
import pytest

from flexrag.common import ChatTurn
from flexrag.processors.rankers import (
    HFRanker,
    HFRankerConfig,
    LiteLLMRanker,
    LiteLLMRankerConfig,
    RankGPTRanker,
    RankGPTRankerConfig,
    RankingResult,
)
from flexrag.resources.invocations import RankerInvocation
from flexrag.resources.runtime_adapters import RemoteRuntimeAdapter

QUERY = "What is the capital of China?"
CANDIDATES = [
    "Shanghai is the largest city in China.",
    "The capital of China is Beijing.",
]


class FakeScorer:
    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []
        return

    def score(self, pairs):
        self.calls.append(pairs)
        return np.array([0.1, 0.9])

    async def async_score(self, pairs):
        return self.score(pairs)


class FakeGenerator:
    def __init__(self, response: str = "2 1") -> None:
        self.response = response
        self.calls = []
        return

    def chat(self, messages, generation_config=None):
        self.calls.append(messages)
        return [[ChatTurn(role="assistant", content=self.response)]]

    async def async_chat(self, messages, generation_config=None):
        return self.chat(messages, generation_config=generation_config)

    def generate(self, prefixes, generation_config=None):
        raise NotImplementedError

    async def async_generate(self, prefixes, generation_config=None):
        raise NotImplementedError


def assert_same_result(left: RankingResult, right: RankingResult) -> None:
    assert left.query == right.query
    assert left.candidates == right.candidates
    assert left.scores == right.scores


def assert_ranked_descending(result: RankingResult) -> None:
    assert result.query == QUERY
    assert sorted(result.candidates) == sorted(CANDIDATES)
    assert result.scores is not None
    assert result.scores == sorted(result.scores, reverse=True)


@pytest.mark.asyncio
async def test_hf_ranker_uses_injected_scorer_sync_and_async():
    scorer = FakeScorer()
    ranker = HFRanker(HFRankerConfig(), scorer=scorer)

    sync_result = ranker.rank(QUERY, CANDIDATES)
    async_result = await ranker.async_rank(QUERY, CANDIDATES)

    assert sync_result.candidates == [CANDIDATES[1], CANDIDATES[0]]
    assert sync_result.scores == [0.9, 0.1]
    assert_same_result(sync_result, async_result)
    assert scorer.calls == [
        [(QUERY, CANDIDATES[0]), (QUERY, CANDIDATES[1])],
        [(QUERY, CANDIDATES[0]), (QUERY, CANDIDATES[1])],
    ]


@pytest.mark.asyncio
async def test_rank_gpt_uses_injected_generator_sync_and_async():
    generator = FakeGenerator(response="2 1")
    ranker = RankGPTRanker(
        RankGPTRankerConfig(window_size=2, step_size=1),
        generator=generator,
    )

    sync_result = ranker.rank(QUERY, CANDIDATES)
    async_result = await ranker.async_rank(QUERY, CANDIDATES)

    assert sync_result.candidates == [CANDIDATES[1], CANDIDATES[0]]
    assert sync_result.scores is None
    assert_same_result(sync_result, async_result)
    assert len(generator.calls) == 2


def test_litellm_ranker_direct_sync_rank(mock_litellm_client):
    ranker = LiteLLMRanker(
        LiteLLMRankerConfig(
            provider="cohere",
            model_name="rerank-v3.5",
            api_key="test",
        )
    )

    result = ranker.rank(QUERY, CANDIDATES)
    call = mock_litellm_client["calls"]["arerank"][0]

    assert_ranked_descending(result)
    assert call["model"] == "cohere/rerank-v3.5"
    assert call["query"] == QUERY
    assert call["documents"] == CANDIDATES
    assert call["top_n"] == len(CANDIDATES)
    assert not call["return_documents"]
    assert call["kwargs"]["api_key"] == "test"


@pytest.mark.asyncio
async def test_litellm_ranker_direct_async_rank(mock_litellm_client):
    ranker = LiteLLMRanker(
        LiteLLMRankerConfig(
            provider="cohere",
            model_name="rerank-v3.5",
            api_key="test",
        )
    )

    result = await ranker.async_rank(QUERY, CANDIDATES)

    assert_ranked_descending(result)
    assert len(mock_litellm_client["calls"]["arerank"]) == 1


@pytest.mark.asyncio
async def test_remote_ranker_runtime_adapter_wraps_litellm_ranker(mock_litellm_client):
    runtime = RemoteRuntimeAdapter(
        LiteLLMRankerConfig(
            provider="cohere",
            model_name="rerank-v3.5",
            api_key="test",
        ),
        impl_cls=LiteLLMRanker,
        max_concurrency=1,
    )
    adapter = RankerInvocation(runtime)

    try:
        sync_result = adapter.rank(QUERY, CANDIDATES)
        async_result = await adapter.async_rank(QUERY, CANDIDATES)
    finally:
        runtime.close()

    assert_ranked_descending(sync_result)
    assert_same_result(sync_result, async_result)
    assert len(mock_litellm_client["calls"]["arerank"]) == 2
