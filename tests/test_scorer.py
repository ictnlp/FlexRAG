import numpy as np
import pytest

from flexrag.models.scorers import (
    HFColBertScorer,
    HFColBertScorerConfig,
    HFCrossEncoderScorer,
    HFCrossEncoderScorerConfig,
    HFLogitsScorer,
    HFLogitsScorerConfig,
)


class TestScorer:
    pairs = [
        ("Who is Bruce Wayne?", "Bruce Wayne is Batman."),
        ("What is the capital of China?", "Beijing is the capital of China."),
        ("Who is Bruce Wayne?", "Thomas Wayne is Bruce Wayne's father."),
    ]

    async def run_scorer(self, scorer) -> None:
        try:
            sync_scores = scorer.score(self.pairs, batch_size=2)
            async_scores = await scorer.async_score(self.pairs, batch_size=2)
            assert isinstance(sync_scores, np.ndarray)
            assert sync_scores.shape == (len(self.pairs),)
            assert async_scores.shape == (len(self.pairs),)
            assert np.max(np.abs(sync_scores - async_scores)) < 1e-4
        finally:
            close = getattr(scorer, "close", None)
            if callable(close):
                close()

    @pytest.mark.asyncio
    async def test_hf_cross_encoder(self):
        scorer = HFCrossEncoderScorer(
            HFCrossEncoderScorerConfig(model_path="cross-encoder/ms-marco-MiniLM-L6-v2")
        )
        await self.run_scorer(scorer)

    @pytest.mark.asyncio
    async def test_hf_logits(self):
        scorer = HFLogitsScorer(
            HFLogitsScorerConfig(model_path="unicamp-dl/InRanker-small")
        )
        await self.run_scorer(scorer)

    @pytest.mark.asyncio
    async def test_hf_colbert(self):
        scorer = HFColBertScorer(
            HFColBertScorerConfig(model_path="colbert-ir/colbertv2.0")
        )
        await self.run_scorer(scorer)
