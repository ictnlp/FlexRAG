from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from flexrag.common import ChatMessages, ChatTurn
from flexrag.common.dataclasses import Context
from flexrag.resources.handles import EncoderHandle
from flexrag.retrievers.backends import Hit

IMPORT_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
IMPORT_ZE_AFFINITY_MASK = os.environ.get("ZE_AFFINITY_MASK")


@dataclass
class FakeEncoderConfig:
    fail_on: str | None = None
    delay_seconds: float = 0.0


def _get(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _texts(inputs: Any) -> list[str]:
    if isinstance(inputs, str):
        return [inputs]
    values = inputs if isinstance(inputs, list) else [inputs]
    return [
        item.get("text", item) if isinstance(item, dict) else item for item in values
    ]


class FakeEncoder:
    def __init__(self, config: Any) -> None:
        self.fail_on = _get(config, "fail_on")
        self.delay_seconds = float(_get(config, "delay_seconds", 0))
        self.pid = os.getpid()

    @property
    def embedding_size(self) -> int:
        return 2

    def encode(self, inputs: Any, *, batch_size: int | None = None) -> np.ndarray:
        del batch_size
        if self.delay_seconds:
            import time

            time.sleep(self.delay_seconds)
        return self._vectorize(inputs)

    def _vectorize(self, inputs: Any) -> np.ndarray:
        vectors = []
        for text in _texts(inputs):
            if text == self.fail_on:
                raise ValueError(f"FakeEncoder failed on {text!r}.")
            vectors.append([float(len(text)), float(sum(map(ord, text)) % 101)])
        return np.array(vectors, dtype=np.float32)

    async def async_encode(
        self,
        inputs: Any,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        del batch_size
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self._vectorize(inputs)

    def runtime_pid(self) -> int:
        return self.pid

    def runtime_env(self) -> dict[str, str | None]:
        return {
            "import_cuda_visible_devices": IMPORT_CUDA_VISIBLE_DEVICES,
            "import_ze_affinity_mask": IMPORT_ZE_AFFINITY_MASK,
            "runtime_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "runtime_ze_affinity_mask": os.environ.get("ZE_AFFINITY_MASK"),
        }


class FakeAsyncEncoder(FakeEncoder):
    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self.remaining_failures = int(_get(config, "failures_before_success", 0))
        self.sync_calls = 0
        self.async_calls = 0

    def encode(self, inputs: Any, *, batch_size: int | None = None) -> np.ndarray:
        del inputs, batch_size
        self.sync_calls += 1
        raise AssertionError("AsyncTarget should use async_encode for sync encode.")

    async def async_encode(
        self,
        inputs: Any,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        self.async_calls += 1
        if self.remaining_failures:
            self.remaining_failures -= 1
            raise ValueError("transient async failure")
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        return self._vectorize(inputs)

    def call_counts(self) -> dict[str, int]:
        return {"sync": self.sync_calls, "async": self.async_calls}


class FakeGenerator:
    def __init__(self, config: Any) -> None:
        self.chat_response = _get(config, "chat_response")

    def chat(
        self, messages: list[ChatMessages], generation_config=None, *, batch_size=None
    ):
        del generation_config, batch_size
        return [
            [ChatTurn(role="assistant", content=self._content(message))]
            for message in messages
        ]

    def _content(self, message: ChatMessages) -> str:
        if self.chat_response is not None:
            return self.chat_response
        last_turn = message.history[-1] if message.history else None
        return "" if last_turn is None else f"chat:{last_turn.text_content}"


class FakeCollectionBackend:
    def __init__(self, config: Any, *, encoders: dict[str, EncoderHandle]) -> None:
        self.encoder = encoders["primary"]
        self.pid = os.getpid()
        self.rows: list[tuple[Context, np.ndarray]] = []
        if _get(config, "startup", False):
            self.startup = {
                "pid": self.pid,
                "embedding_size": self.encoder.embedding_size,
                "vector": self.encoder.encode("startup")[0].tolist(),
            }

    def rebuild(self, contexts: list[Context]) -> None:
        texts = [context.data["text"] for context in contexts]
        self.rows = list(zip(contexts, self.encoder.encode(texts), strict=True))

    def search_hits(self, queries: list[str], top_k: int, *, search_options=None):
        del search_options
        hits = [
            Hit(
                context_id=ctx.context_id, score=float(idx), backend="fake", view="text"
            )
            for idx, (ctx, _) in enumerate(self.rows[:top_k])
        ]
        return [hits for _ in queries]

    def count(self) -> int:
        return len(self.rows)

    def runtime_pid(self) -> int:
        return self.pid
