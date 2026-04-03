import asyncio
import time

import pytest

from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models.generators import GenerationConfig
from tests.support.process_test_support import (
    FakeLocalGenerator,
    FakeLocalGeneratorConfig,
)


def expected_generated(prefixes: list[str], sample_num: int = 1) -> list[list[str]]:
    return [
        [f"{prefix} -> sample {i}" for i in range(sample_num)] for prefix in prefixes
    ]


def expected_chat(
    messages: list[ChatMessages], sample_num: int = 1
) -> list[list[ChatTurn]]:
    prompts = [" ".join(turn.text_content for turn in message) for message in messages]
    return [
        [
            ChatTurn(role="assistant", content=f"{prompt} -> reply {i}")
            for i in range(sample_num)
        ]
        for prompt in prompts
    ]


@pytest.mark.asyncio
async def test_local_process_generator_sync_async_consistency():
    prefixes = ["alpha", "bravo", "charlie"]
    messages = [
        ChatMessages(history=[ChatTurn(role="user", content=prefix)])
        for prefix in prefixes
    ]
    cfg = GenerationConfig(sample_num=2, do_sample=True)
    with FakeLocalGenerator(FakeLocalGeneratorConfig()) as generator:
        sync_generated = generator.generate(prefixes, cfg)
        async_generated = await generator.async_generate(prefixes, cfg)
        assert sync_generated == expected_generated(prefixes, sample_num=2)
        assert async_generated == sync_generated

        sync_chat = generator.chat(messages, cfg)
        async_chat = await generator.async_chat(messages, cfg)
        assert sync_chat == expected_chat(messages, sample_num=2)
        assert async_chat == sync_chat


@pytest.mark.asyncio
async def test_local_process_generator_batch_scheduling():
    prefixes = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    with FakeLocalGenerator(
        FakeLocalGeneratorConfig(device_id=[0, 1, 2], delay_s=0.2)
    ) as generator:
        await generator.async_generate(
            ["warmup-1", "warmup-2", "warmup-3", "warmup-4", "warmup-5", "warmup-6"],
            batch_size=2,
        )
        start = time.perf_counter()
        outputs = await generator.async_generate(prefixes, batch_size=2)
        elapsed = time.perf_counter() - start

        assert outputs == expected_generated(prefixes)
        assert elapsed < 0.55


@pytest.mark.asyncio
async def test_local_process_generator_async_does_not_block_loop():
    generator = FakeLocalGenerator(FakeLocalGeneratorConfig(delay_s=0.2))
    try:
        task = asyncio.create_task(generator.async_generate(["slow"]))
        await asyncio.sleep(0.05)
        assert not task.done()
        await asyncio.sleep(0.01)
        result = await task
        assert result == expected_generated(["slow"])
    finally:
        generator.close()


def test_local_process_generator_context_manager_closes_workers():
    generator = FakeLocalGenerator(FakeLocalGeneratorConfig(device_id=[0, 1]))
    with generator:
        generator.generate(["alpha", "bravo"])
        client = generator._client
        assert client is not None
        assert all(worker.process.is_alive() for worker in client._workers)
    assert all(not worker.process.is_alive() for worker in client._workers)


@pytest.mark.asyncio
async def test_local_process_generator_propagates_worker_errors():
    generator = FakeLocalGenerator(FakeLocalGeneratorConfig(error_on="boom"))
    try:
        with pytest.raises(RuntimeError, match="ValueError: boom: boom"):
            await generator.async_generate(["boom"])
    finally:
        generator.close()
