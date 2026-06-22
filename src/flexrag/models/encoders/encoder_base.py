import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias, cast

import numpy as np
from PIL.ImageFile import ImageFile

from flexrag.common import ContentPart, Register

EncoderInput: TypeAlias = str | ImageFile | ContentPart
EncoderInputs: TypeAlias = EncoderInput | list[EncoderInput]


def _normalize_encoder_inputs(inputs: EncoderInputs) -> list[ContentPart]:
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


class EncoderProtocol(Protocol):
    """Protocol for directly usable raw encoders.

    Raw encoders expose a common public interface for direct use. The public
    methods normalize convenient input shapes to content blocks before
    dispatch. Implementations do not provide runtime policies such as
    deployment batching, progress logging, process isolation, retry, or rate
    limiting.
    """

    def encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        """
        ...

    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items asynchronously.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        """
        ...

    @property
    def embedding_size(self) -> int | None:
        """Return the embedding dimension when it is known."""
        ...


class LocalEncoderBase(ABC):
    """Thin base class for directly usable local encoders.

    Subclasses implement synchronous canonical-batch ``_encode_batch``. The
    public ``encode`` method normalizes convenient input shapes to content
    blocks, splits direct-use calls according to ``batch_size``, and merges the
    resulting arrays. The async method is a convenience wrapper built with
    ``asyncio.to_thread``; it keeps an event loop responsive but does not
    provide process isolation, retry, rate limiting, progress logging, or true
    Python-level parallelism.
    """

    def __init__(self, batch_size: int = 32) -> None:
        """Initialize direct-use local encoder batching.

        :param batch_size: Maximum batch size used by the raw local encoder's
            public ``encode`` method.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.batch_size = batch_size
        return

    def encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        """
        normalized_inputs = _normalize_encoder_inputs(inputs)
        if not normalized_inputs:
            embedding_size = self.embedding_size
            if embedding_size is None:
                return np.array([])
            return np.empty((0, embedding_size), dtype=np.float32)

        results = [
            self._encode_batch(normalized_inputs[i : i + self.batch_size])
            for i in range(0, len(normalized_inputs), self.batch_size)
        ]
        if len(results) == 1:
            return results[0]
        return np.concatenate(results, axis=0)

    @abstractmethod
    def _encode_batch(self, inputs: list[ContentPart]) -> np.ndarray:
        """Encode one implementation batch.

        :param inputs: Canonical content-part batch accepted by the concrete
            encoder implementation.
        :return: One embedding row for each input item.
        """
        return

    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items asynchronously.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        """
        return await asyncio.to_thread(self.encode, inputs)

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        """Return the embedding dimension when it is known."""
        return


class RemoteEncoderBase(ABC):
    """Thin base class for directly usable remote encoders.

    Subclasses implement asynchronous canonical-batch ``_async_encode_batch``.
    The public async method normalizes convenient input shapes to content
    blocks. The synchronous method runs the async method with ``asyncio.run``
    and must not be called from an already running event loop.
    """

    @staticmethod
    def _ensure_sync_bridge_allowed(method_name: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            f"{method_name} cannot be called from a running event loop. "
            f"Use async_{method_name} instead."
        )

    def encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items synchronously.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("encode")
        return asyncio.run(self.async_encode(inputs))

    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one input item or a batch of input items asynchronously.

        :param inputs: A string, image content, content block, or a list of
            those values.
        :return: One embedding row for each input item.
        """
        return await self._async_encode_batch(_normalize_encoder_inputs(inputs))

    @abstractmethod
    async def _async_encode_batch(self, inputs: list[ContentPart]) -> np.ndarray:
        """Encode one canonical content-part batch asynchronously.

        :param inputs: Canonical content-part batch accepted by the concrete
            encoder implementation.
        :return: One embedding row for each input item.
        """
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        """Return the embedding dimension when it is known."""
        return


ENCODERS = Register[EncoderProtocol]("encoder")
