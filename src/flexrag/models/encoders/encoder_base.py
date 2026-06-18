import asyncio
from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias

import numpy as np

from flexrag.common import ContentPart, Register

EncoderInputs: TypeAlias = list[str] | list[ContentPart]


class EncoderProtocol(Protocol):
    """Protocol for directly usable raw encoders.

    Raw encoders expose a common canonical-batch interface for direct use.
    Implementations do not provide runtime policies such as deployment
    batching, progress logging, process isolation, retry, or rate limiting.
    """

    def encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode a batch of canonical encoder inputs.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
        :return: One embedding row for each input item.
        """
        ...

    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode a batch of canonical encoder inputs asynchronously.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
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
    public ``encode`` method splits direct-use calls according to
    ``batch_size`` and merges the resulting arrays. The async method is a
    convenience wrapper built with ``asyncio.to_thread``; it keeps an event loop
    responsive but does not provide process isolation, retry, rate limiting,
    progress logging, or true Python-level parallelism.
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
        """Encode a batch of canonical encoder inputs.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
        :return: One embedding row for each input item.
        """
        if not inputs:
            embedding_size = self.embedding_size
            if embedding_size is None:
                return np.array([])
            return np.empty((0, embedding_size), dtype=np.float32)

        results = [
            self._encode_batch(inputs[i : i + self.batch_size])
            for i in range(0, len(inputs), self.batch_size)
        ]
        if len(results) == 1:
            return results[0]
        return np.concatenate(results, axis=0)

    @abstractmethod
    def _encode_batch(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode one implementation batch.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
        :return: One embedding row for each input item.
        """
        return

    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode a batch of canonical encoder inputs asynchronously.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
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

    Subclasses implement asynchronous canonical-batch ``async_encode``. The
    synchronous method runs the async method with ``asyncio.run`` and must not
    be called from an already running event loop.
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
        """Encode a batch of canonical encoder inputs synchronously.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
        :return: One embedding row for each input item.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("encode")
        return asyncio.run(self.async_encode(inputs))

    @abstractmethod
    async def async_encode(self, inputs: EncoderInputs) -> np.ndarray:
        """Encode a batch of canonical encoder inputs asynchronously.

        :param inputs: Canonical text or content-part batch accepted by the
            concrete encoder implementation.
        :return: One embedding row for each input item.
        """
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int | None:
        """Return the embedding dimension when it is known."""
        return


ENCODERS = Register[EncoderProtocol]("encoder")
