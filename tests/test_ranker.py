import os
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

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
from flexrag.utils import LOGGER_MANAGER, ConfigureBase

logger = LOGGER_MANAGER.get_logger("tests.test_ranker")


@dataclass
class RankerTestConfig(ConfigureBase):
    cohere_configs: list[CohereRankerConfig] = field(default_factory=list)
    jina_configs: list[JinaRankerConfig] = field(default_factory=list)
    mixedbread_configs: list[MixedbreadRankerConfig] = field(default_factory=list)
    voyage_configs: list[VoyageRankerConfig] = field(default_factory=list)
    rankgpt_configs: list[RankGPTRankerConfig] = field(default_factory=list)
    hf_cross_configs: list[HFCrossEncoderRankerConfig] = field(default_factory=list)
    hf_seq2seq_configs: list[HFSeq2SeqRankerConfig] = field(default_factory=list)
    hf_colbert_configs: list[HFColBertRankerConfig] = field(default_factory=list)


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

    def valid_result(self, r1: RankingResult, r2: RankingResult) -> None:
        for c1, c2 in zip(r1.candidates, r2.candidates):
            assert c1 == c2
        if r1.scores is not None:
            for s1, s2 in zip(r1.scores, r2.scores):
                assert s1 - s2 < 1e-4
        return

    @pytest.mark.asyncio
    async def test_rank_cohere(self):
        for cfg in self.cfg.cohere_configs:
            logger.debug(f"Testing CohereRanker({cfg.model}).")
            ranker = CohereRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing CohereRanker({cfg.model}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_jina(self):
        for cfg in self.cfg.jina_configs:
            logger.debug(f"Testing JinaRanker({cfg.model}).")
            ranker = JinaRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing JinaRanker({cfg.model}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_mixedbread(self):
        for cfg in self.cfg.mixedbread_configs:
            logger.debug(f"Testing MixedbreadRanker({cfg.model}).")
            ranker = MixedbreadRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing MixedbreadRanker({cfg.model}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_voyage(self):
        for cfg in self.cfg.voyage_configs:
            logger.debug(f"Testing VoyageRanker({cfg.model}).")
            ranker = VoyageRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing VoyageRanker({cfg.model}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_gpt(self):
        for cfg in self.cfg.rankgpt_configs:
            logger.debug(f"Testing RankGPTRanker({cfg.generator_type}).")
            ranker = RankGPTRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing RankGPTRanker({cfg.generator_type}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_hf_cross(self):
        for cfg in self.cfg.hf_cross_configs:
            logger.debug(f"Testing HFCrossEncoderRanker({cfg.model_path}).")
            ranker = HFCrossEncoderRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing HFCrossEncoderRanker({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_hf_seq2seq(self):
        for cfg in self.cfg.hf_seq2seq_configs:
            logger.debug(f"Testing HFSeq2SeqRanker({cfg.model_path}).")
            ranker = HFSeq2SeqRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing HFSeq2SeqRanker({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_rank_hf_colbert(self):
        for cfg in self.cfg.hf_colbert_configs:
            logger.debug(f"Testing HFColBertRanker({cfg.model_path}).")
            ranker = HFColBertRanker(cfg)
            r1 = ranker.rank(self.query, self.candidates)
            r2 = await ranker.async_rank(self.query, self.candidates)
            self.valid_result(r1, r2)
            logger.debug(f"Testing HFColBertRanker({cfg.model_path}) done.")
        return
