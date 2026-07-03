import asyncio
import time

import numpy as np
import pytest

from tests.support.process.process_test_support import (
    FakeLocalPairScorer,
    FakeLocalPairScorerConfig,
)


def expected_scores(pairs: list[tuple[str, str]]) -> np.ndarray:
    scores = []
    for query, candidate in pairs:
        checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(query + candidate))
        scores.append(
            float((len(query) * 3 + len(candidate) * 5 + checksum) % 997) / 997.0
        )
    return np.array(scores, dtype=np.float32)


@pytest.mark.asyncio
async def test_local_process_scorer_sync_async_consistency():
    pairs = [("alpha", "one"), ("bravo", "two"), ("charlie", "three")]
    with FakeLocalPairScorer(FakeLocalPairScorerConfig()) as scorer:
        sync_scores = scorer.score(pairs)
        async_scores = await scorer.async_score(pairs)

        assert isinstance(sync_scores, np.ndarray)
        assert sync_scores.shape == (len(pairs),)
        assert np.array_equal(sync_scores, expected_scores(pairs))
        assert np.array_equal(sync_scores, async_scores)


@pytest.mark.asyncio
async def test_local_process_scorer_batch_scheduling():
    pairs = [
        ("alpha", "one"),
        ("bravo", "two"),
        ("charlie", "three"),
        ("delta", "four"),
        ("echo", "five"),
        ("foxtrot", "six"),
    ]
    with FakeLocalPairScorer(
        FakeLocalPairScorerConfig(delay_s=0.2),
        batch_size=2,
        device_groups=[[0], [1], [2]],
    ) as scorer:
        await scorer.async_score(
            [
                ("warmup-1", "candidate"),
                ("warmup-2", "candidate"),
                ("warmup-3", "candidate"),
                ("warmup-4", "candidate"),
                ("warmup-5", "candidate"),
                ("warmup-6", "candidate"),
            ],
        )
        start = time.perf_counter()
        scores = await scorer.async_score(pairs)
        elapsed = time.perf_counter() - start

        assert np.array_equal(scores, expected_scores(pairs))
        assert elapsed < 0.55


@pytest.mark.asyncio
async def test_local_process_scorer_async_does_not_block_loop():
    scorer = FakeLocalPairScorer(FakeLocalPairScorerConfig(delay_s=0.2))
    try:
        task = asyncio.create_task(scorer.async_score([("slow", "candidate")]))
        await asyncio.sleep(0.05)
        assert not task.done()
        await asyncio.sleep(0.01)
        scores = await task
        assert np.array_equal(scores, expected_scores([("slow", "candidate")]))
    finally:
        scorer.close()


def test_local_process_scorer_context_manager_closes_workers():
    scorer = FakeLocalPairScorer(
        FakeLocalPairScorerConfig(),
        device_groups=[[0], [1]],
    )
    with scorer:
        scorer.score([("alpha", "one"), ("bravo", "two")])
        client = scorer._client
        assert client is not None
        assert all(worker.process.is_alive() for worker in client._workers)
    assert all(not worker.process.is_alive() for worker in client._workers)


@pytest.mark.asyncio
async def test_local_process_scorer_propagates_worker_errors():
    scorer = FakeLocalPairScorer(FakeLocalPairScorerConfig(error_on="boom"))
    try:
        with pytest.raises(RuntimeError, match="ValueError: boom: boom"):
            await scorer.async_score([("boom", "candidate")])
    finally:
        scorer.close()
