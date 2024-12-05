import os
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from librarian.ranker import (
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
    VoyageRanker,
    VoyageRankerConfig,
)


@dataclass
class RankerTestConfig:
    cohere_config: CohereRankerConfig = field(default_factory=CohereRankerConfig)
    jina_config: JinaRankerConfig = field(default_factory=JinaRankerConfig)
    mixedbread_config: MixedbreadRankerConfig = field(default_factory=MixedbreadRankerConfig)  # fmt: skip
    voyage_config: VoyageRankerConfig = field(default_factory=VoyageRankerConfig)
    rankgpt_config: RankGPTRankerConfig = field(default_factory=RankGPTRankerConfig)
    hf_cross_config: HFCrossEncoderRankerConfig = field(default_factory=HFCrossEncoderRankerConfig)  # fmt: skip
    hf_seq2seq_config: HFSeq2SeqRankerConfig = field(default_factory=HFSeq2SeqRankerConfig)  # fmt: skip
    hf_colbert_config: HFColBertRankerConfig = field(default_factory=HFColBertRankerConfig)  # fmt: skip


class TestRanker:
    cfg: RankerTestConfig = OmegaConf.merge(
        OmegaConf.structured(RankerTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "ranker.yaml")
        ),
    )
    query = "What is the capital of China?"
    candidates = [
        "The capital of China is Beijing.",
        "Shanghai is the largest city in China.",
    ]

    def valid_result(self, r1, r2): ...

    @pytest.mark.asyncio
    async def test_rank_cohere(self):
        ranker = CohereRanker(self.cfg.cohere_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_jina(self):
        ranker = JinaRanker(self.cfg.jina_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_mixedbread(self):
        ranker = MixedbreadRanker(self.cfg.mixedbread_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_voyage(self):
        ranker = VoyageRanker(self.cfg.voyage_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_gpt(self):
        ranker = RankGPTRanker(self.cfg.rankgpt_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_cross(self):
        ranker = HFCrossEncoderRanker(self.cfg.hf_cross_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_seq2seq(self):
        ranker = HFSeq2SeqRanker(self.cfg.hf_seq2seq_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return

    @pytest.mark.asyncio
    async def test_rank_hf_colbert(self):
        ranker = HFColBertRanker(self.cfg.hf_colbert_config)
        r1 = ranker.rank(self.query, self.candidates)
        r2 = await ranker.async_rank(self.query, self.candidates)
        self.valid_result(r1, r2)
        return
