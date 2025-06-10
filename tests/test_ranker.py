import pytest

from flexrag.models import OpenAIGeneratorConfig
from flexrag.ranker import (
    CohereRanker,
    CohereRankerConfig,
    HFColBertRanker,
    HFColBertRankerConfig,
    HFCrossEncoderRanker,
    HFCrossEncoderRankerConfig,
    HFSeq2SeqRanker,
    HFSeq2SeqRankerConfig,
    JinaRanker,
    JinaRankerConfig,
    MixedbreadRanker,
    MixedbreadRankerConfig,
    RankGPTRanker,
    RankGPTRankerConfig,
    RankingResult,
    VoyageRanker,
    VoyageRankerConfig,
)
from flexrag.utils import LOGGER_MANAGER

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
    async def test_rank_cohere(self, mock_cohere_client):
        ranker = CohereRanker(CohereRankerConfig(api_key="test"))
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_jina(self, mock_jina_client):
        ranker = JinaRanker(JinaRankerConfig(api_key="test"))
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_mixedbread(self, mock_mixedbread_client):
        ranker = MixedbreadRanker(MixedbreadRankerConfig(api_key="test"))
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_voyage(self, mock_voyage_client):
        ranker = VoyageRanker(VoyageRankerConfig(api_key="test"))
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_rank_gpt(self, mock_openai_client):
        ranker = RankGPTRanker(
            RankGPTRankerConfig(
                generator_type="openai",
                openai_config=OpenAIGeneratorConfig(
                    model_name="gpt-4",
                    api_key="test",
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
        ranker = HFSeq2SeqRanker(
            HFSeq2SeqRankerConfig(model_path="unicamp-dl/InRanker-small")
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
