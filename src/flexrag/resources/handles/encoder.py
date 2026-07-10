from __future__ import annotations

import numpy as np

from flexrag.models.encoders.encoder_base import (
    EncoderInputs,
    _normalize_encoder_inputs,
)

from ..runtime import RuntimeCall
from .base import TypedHandle


class EncoderHandle(TypedHandle):
    """Typed proxy for encoder resources.

    The handle normalizes formal encoder inputs, splits them using the target
    batch size, submits primitive runtime calls, and merges array results. It
    does not own the encoder lifecycle.
    """

    def encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> np.ndarray | list[list[float]]:
        """Synchronously encode inputs.

        :param inputs: Encoder inputs accepted by the formal encoder API.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Encoded vectors in input order.
        """
        normalized = _normalize_encoder_inputs(inputs)
        if not normalized:
            embedding_size = self.embedding_size
            if embedding_size is None:
                return np.array([])
            return np.empty((0, embedding_size), dtype=np.float32)

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "encode",
                args=(batch,),
                kwargs={"batch_size": batch_size},
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results = self._target.batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Encoding",
        )
        return self._merge_arrays(results)

    async def async_encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> np.ndarray | list[list[float]]:
        """Asynchronously encode inputs.

        :param inputs: Encoder inputs accepted by the formal encoder API.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Encoded vectors in input order.
        """
        normalized = _normalize_encoder_inputs(inputs)
        if not normalized:
            embedding_size = self.embedding_size
            if embedding_size is None:
                return np.array([])
            return np.empty((0, embedding_size), dtype=np.float32)

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "async_encode",
                args=(batch,),
                kwargs={"batch_size": batch_size},
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results = await self._target.async_batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Encoding",
        )
        return self._merge_arrays(results)

    @property
    def embedding_size(self) -> int | None:
        """Return the encoder embedding size when the raw resource exposes it."""
        try:
            return self._target.getattr("embedding_size")
        except AttributeError:
            return None

    @staticmethod
    def _merge_arrays(results: list[np.ndarray]) -> np.ndarray:
        if not results:
            return np.array([])
        arrays = [np.asarray(result) for result in results]
        if len(arrays) == 1:
            return arrays[0]
        return np.concatenate(arrays, axis=0)
