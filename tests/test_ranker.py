import pytest

from flexrag.common import LOGGER_MANAGER
from flexrag.models import LiteLLMGeneratorConfig
from flexrag.processors.rankers import (
    HFColBertRanker,
    HFColBertRankerConfig,
    HFCrossEncoderRanker,
    HFCrossEncoderRankerConfig,
    HFLogitsRanker,
    HFLogitsRankerConfig,
    LiteLLMRanker,
    LiteLLMRankerConfig,
    RankGPTRanker,
    RankGPTRankerConfig,
    RankingResult,
)

logger = LOGGER_MANAGER.get_logger("tests.test_ranker")


class TestRanker:
    query = "What is the capital of China?"
    candidates = [
        "The capital of China is Beijing.",
        "Shanghai is the largest city in China.",
    ]

    def valid_result(self, r1: RankingResult, r2: RankingResult) -> None:
        for c1, c2 in zip(r1.candidates, r2.candidates):
            assert c1 == c2
        if r1.scores is not None:
            for s1, s2 in zip(r1.scores, r2.scores):
                assert s1 - s2 < 1e-4
        return

    @pytest.mark.asyncio
    async def test_rank_litellm(self, mock_litellm_client):
        ranker = LiteLLMRanker(
            LiteLLMRankerConfig(
                provider="cohere",
                model_name="rerank-v3.5",
                api_key="test",
            )
        )
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_rank_gpt(self, mock_litellm_client):
        ranker = RankGPTRanker(
            RankGPTRankerConfig(
                generator_type="litellm",
                litellm_config=LiteLLMGeneratorConfig(
                    provider="openai",
                    model_name="gpt-4o-mini",
                ),
            )
        )
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_cross(self):
        ranker = HFCrossEncoderRanker(
            HFCrossEncoderRankerConfig(model_path="cross-encoder/ms-marco-MiniLM-L6-v2")
        )
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_seq2seq(self):
        ranker = HFLogitsRanker(
            HFLogitsRankerConfig(model_path="unicamp-dl/InRanker-small")
        )
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_colbert(self):
        ranker = HFColBertRanker(
            HFColBertRankerConfig(model_path="colbert-ir/colbertv2.0")
        )
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return
