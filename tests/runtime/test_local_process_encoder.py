import asyncio
import time

import numpy as np
import pytest

from tests.support.process.process_test_support import (
    FakeLocalTextEncoder,
    FakeLocalTextEncoderConfig,
)


def expected_embeddings(texts: list[str], embedding_dim: int = 3) -> np.ndarray:
    rows = []
    for text in texts:
        checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 997
        rows.append(
            [
                float(len(text)),
                float(sum(ord(ch) for ch in text) % 97),
                float(checksum),
            ][:embedding_dim]
        )
    return np.array(rows, dtype=np.float32)


@pytest.mark.asyncio
async def test_local_process_encoder_sync_async_consistency():
    texts = ["alpha", "bravo", "charlie"]
    with FakeLocalTextEncoder(FakeLocalTextEncoderConfig()) as encoder:
        sync_result = encoder.encode(texts)
        async_result = await encoder.async_encode(texts)

        assert isinstance(sync_result, np.ndarray)
        assert sync_result.shape == (len(texts), encoder.embedding_size)
        assert np.array_equal(sync_result, expected_embeddings(texts))
        assert np.array_equal(sync_result, async_result)


@pytest.mark.asyncio
async def test_local_process_encoder_batch_scheduling():
    texts = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    with FakeLocalTextEncoder(
        FakeLocalTextEncoderConfig(
            delay_s=0.2,
        ),
        batch_size=2,
        device_groups=[[0], [1], [2]],
    ) as encoder:
        await encoder.async_encode(
            ["warmup-1", "warmup-2", "warmup-3", "warmup-4", "warmup-5", "warmup-6"],
        )
        start = time.perf_counter()
        embeddings = await encoder.async_encode(texts)
        elapsed = time.perf_counter() - start

        assert np.array_equal(embeddings, expected_embeddings(texts))
        assert elapsed < 0.55


@pytest.mark.asyncio
async def test_local_process_encoder_async_does_not_block_loop():
    encoder = FakeLocalTextEncoder(FakeLocalTextEncoderConfig(delay_s=0.2))
    try:
        task = asyncio.create_task(encoder.async_encode(["slow"]))
        await asyncio.sleep(0.05)
        assert not task.done()
        await asyncio.sleep(0.01)
        result = await task
        assert np.array_equal(result, expected_embeddings(["slow"]))
    finally:
        encoder.close()


def test_local_process_encoder_context_manager_closes_workers():
    encoder = FakeLocalTextEncoder(
        FakeLocalTextEncoderConfig(),
        device_groups=[[0], [1]],
    )
    with encoder:
        encoder.encode(["alpha", "bravo"])
        client = encoder._client
        assert client is not None
        assert all(worker.process.is_alive() for worker in client._workers)
    assert all(not worker.process.is_alive() for worker in client._workers)


@pytest.mark.asyncio
async def test_local_process_encoder_propagates_worker_errors():
    encoder = FakeLocalTextEncoder(FakeLocalTextEncoderConfig(error_on="boom"))
    try:
        with pytest.raises(RuntimeError, match="ValueError: boom: boom"):
            await encoder.async_encode(["boom"])
    finally:
        encoder.close()
