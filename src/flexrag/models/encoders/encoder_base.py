import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias, cast

import numpy as np
from PIL.ImageFile import ImageFile

from flexrag.common import ContentPart, ProgressDisplay, Register, SimpleProgressLogger
from flexrag.models.async_client_base import AsyncClientMixin, ConfigT

EncoderInput: TypeAlias = str | ImageFile | ContentPart


def normalize_encoder_inputs(
    inputs: EncoderInput | list[EncoderInput],
) -> list[ContentPart]:
    items = inputs if isinstance(inputs, list) else [inputs]
    normalized: list[ContentPart] = []
    for item in items:
        if isinstance(item, str):
            normalized.append({"type": "text", "text": item})
            continue
        if isinstance(item, ImageFile):
            normalized.append({"type": "image", "image": item})
            continue
        if isinstance(item, dict):
            content_type = item.get("type")
            if not isinstance(content_type, str):
                raise ValueError("Encoder content blocks must include a string 'type'.")
            normalized.append(cast(ContentPart, item))
            continue
        raise TypeError(f"Unsupported encoder input type: {type(item).__name__}")
    return normalized


def extract_text_encoder_inputs(
    inputs: EncoderInput | list[EncoderInput],
    *,
    encoder_name: str,
) -> list[str]:
    normalized_inputs = normalize_encoder_inputs(inputs)
    texts: list[str] = []
    for part in normalized_inputs:
        if part.get("type") != "text":
            raise ValueError(
                f"{encoder_name} only supports text content blocks, "
                f"but got '{part.get('type')}'."
            )
        texts.append(part.get("text", ""))
    return texts


class EncoderProtocol(Protocol):
    def encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        batch_size: int = 32,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray: ...

    async def async_encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        batch_size: int = 32,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray: ...

    @property
    def embedding_size(self) -> int | None: ...


class EncoderBase(AsyncClientMixin[ConfigT], ABC):
    """Base class for client-backed encoder implementations."""

    def __init__(self, config: ConfigT):
        AsyncClientMixin.__init__(self, config)
        return

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @abstractmethod
    async def _async_encode_impl(self, client, inputs: list[ContentPart]) -> np.ndarray:
        return

    async def _async_encode_core(
        self,
        inputs: EncoderInput | list[EncoderInput],
        batch_size: int | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        normalized_inputs = normalize_encoder_inputs(inputs)

        if batch_size is None:
            batches = [normalized_inputs]
        else:
            batches = [
                normalized_inputs[i : i + batch_size]
                for i in range(0, len(normalized_inputs), batch_size)
            ]
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(normalized_inputs), interval=log_interval, display=display
        ) as p_logger:

            async def _encode_task(idx: int, batch: list[ContentPart]) -> None:
                async with semaphore:
                    res = await self._async_encode_impl(client, batch)
                results[idx] = res
                p_logger.update(len(batch), desc="Encoding")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _encode_task(idx, batch), name=f"encode_batch_{idx}"
                        )
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        if not results:
            return np.array([])
        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some encode tasks did not produce results.")
        if len(ready_results) == 1:
            return ready_results[0]
        return np.concatenate(ready_results, axis=0)

    async def async_encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        batch_size: int = 32,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return await self._run_coroutine_async(
            self._async_encode_core(
                inputs,
                batch_size=batch_size,
                log_interval=log_interval,
                display=display,
            )
        )

    def encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        batch_size: int = 32,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return self._run_coroutine_sync(
            self._async_encode_core(
                inputs,
                batch_size=batch_size,
                log_interval=log_interval,
                display=display,
            )
        )

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        return


ENCODERS = Register[EncoderProtocol]("encoder")
