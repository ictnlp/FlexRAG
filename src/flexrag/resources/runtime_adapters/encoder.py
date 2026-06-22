import asyncio
from abc import abstractmethod
from typing import Literal, TypeAlias, cast

import numpy as np
from PIL.ImageFile import ImageFile

from flexrag.common import ContentPart, ProgressDisplay, SimpleProgressLogger
from flexrag.models.encoders.encoder_base import LocalEncoderBase, RemoteEncoderBase
from flexrag.runtime.async_client import AsyncClientMixin, ConfigT
from flexrag.runtime.process_worker_pool import ProcessWorkerPoolClient

EncoderInput: TypeAlias = str | ImageFile | ContentPart
EncoderInputFormat: TypeAlias = Literal["content", "text"]


def _normalize_inputs(
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


def _extract_text_inputs(
    inputs: list[ContentPart],
    *,
    encoder_name: str,
) -> list[str]:
    texts: list[str] = []
    for part in inputs:
        if part.get("type") != "text":
            raise ValueError(
                f"{encoder_name} only supports text content blocks, "
                f"but got '{part.get('type')}'."
            )
        texts.append(part.get("text", ""))
    return texts


def _merge_encode_results(results: list[None | np.ndarray]) -> np.ndarray:
    if not results:
        return np.array([])
    ready_results = [result for result in results if result is not None]
    if len(ready_results) != len(results):
        raise RuntimeError("Some encode tasks did not produce results.")
    if len(ready_results) == 1:
        return ready_results[0]
    return np.concatenate(ready_results, axis=0)


class EncoderRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Managed call adapter for encoder-like resources.

    This base adapter provides the public sync and async call surface for
    encoder resources. It normalizes accepted input shapes, submits work to the
    managed background event loop, and delegates the actual encoding work to
    subclass core methods.
    """

    input_format: EncoderInputFormat = "content"

    def __init__(
        self,
        config: ConfigT,
        *,
        input_format: EncoderInputFormat | None = None,
    ) -> None:
        super().__init__(config)
        if input_format is not None:
            self.input_format = input_format
        return

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    def _prepare_encoder_inputs(
        self,
        inputs: list[ContentPart],
    ) -> list[str] | list[ContentPart]:
        if self.input_format == "text":
            return _extract_text_inputs(
                inputs,
                encoder_name=self.__class__.__name__,
            )
        return inputs

    @abstractmethod
    async def _async_encode_core(
        self,
        inputs: list[ContentPart],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return

    async def async_encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Encode inputs asynchronously.

        ``inputs`` may be a single string, image object, content block, or a
        list of those values. The adapter normalizes it to content blocks before
        dispatch and returns one embedding row per input item.

        :param inputs: Input item or batch to encode.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Encoded embeddings.
        """
        normalized_inputs = _normalize_inputs(inputs)
        return await self._run_coroutine_async(
            self._async_encode_core(
                normalized_inputs,
                log_interval=log_interval,
                display=display,
            )
        )

    def encode(
        self,
        inputs: EncoderInput | list[EncoderInput],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        """Encode inputs synchronously.

        ``inputs`` may be a single string, image object, content block, or a
        list of those values. The adapter normalizes it to content blocks and
        runs the async encoding core on the managed background event loop.

        :param inputs: Input item or batch to encode.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Encoded embeddings.
        """
        normalized_inputs = _normalize_inputs(inputs)
        return self._run_coroutine_sync(
            self._async_encode_core(
                normalized_inputs,
                log_interval=log_interval,
                display=display,
            )
        )

    async def _async_embedding_size(self) -> int | None:
        client = await self._get_async_client()
        return getattr(client, "embedding_size", None)

    @property
    def embedding_size(self) -> int | None:
        return self._run_coroutine_sync(self._async_embedding_size())


class RemoteEncoderRuntimeAdapter(EncoderRuntimeAdapter[ConfigT]):
    """Managed runtime adapter for remote raw encoders.

    Remote raw encoders expose asynchronous canonical-batch ``async_encode``.
    This adapter splits managed calls into deployment batches while preserving
    result order. It also owns remote-runtime policies such as maximum
    concurrency, requests-per-minute limiting, retry delays, and progress
    logging.
    """

    impl_cls: type[RemoteEncoderBase] | None = None

    def __init__(
        self,
        config: ConfigT,
        impl_cls: type[RemoteEncoderBase] | None = None,
        *,
        input_format: EncoderInputFormat | None = None,
        batch_size: int = 32,
        max_concurrency: int = 1,
        rpm: float = 0,
        retry_times: int = 0,
        retry_min_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        """Create a remote encoder runtime adapter.

        :param config: Configuration passed to the raw encoder implementation.
        :param impl_cls: Optional raw encoder implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param input_format: Canonical input format expected by the raw encoder.
            ``"text"`` extracts text strings from text content blocks.
        :param batch_size: Deployment batch size used for remote requests.
        :param max_concurrency: Maximum number of in-flight batch requests.
        :param rpm: Requests-per-minute limit. ``0`` disables rate limiting.
        :param retry_times: Number of retries after the initial attempt. ``0``
            disables retry.
        :param retry_min_delay: Initial retry delay in seconds.
        :param retry_max_delay: Maximum retry delay in seconds.
        :raises ValueError: If any runtime policy value is invalid.
        """
        super().__init__(config, input_format=input_format)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0.")
        if rpm < 0:
            raise ValueError("rpm must be non-negative.")
        if retry_times < 0:
            raise ValueError("retry_times must be non-negative.")
        if retry_min_delay < 0:
            raise ValueError("retry_min_delay must be non-negative.")
        if retry_max_delay < retry_min_delay:
            raise ValueError(
                "retry_max_delay must be greater than or equal to retry_min_delay."
            )
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._rpm = rpm
        self._retry_times = retry_times
        self._retry_min_delay = retry_min_delay
        self._retry_max_delay = retry_max_delay
        self._rpm_lock: asyncio.Lock | None = None
        self._next_request_time: float = 0.0
        return

    async def _create_client(self, config: ConfigT):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        return self.impl_cls(config)

    def _get_max_concurrency(self) -> int:
        return self._max_concurrency

    async def _wait_for_rpm(self) -> None:
        if self._rpm <= 0:
            return
        if self._rpm_lock is None:
            self._rpm_lock = asyncio.Lock()
        interval = 60.0 / self._rpm
        async with self._rpm_lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_request_time > now:
                await asyncio.sleep(self._next_request_time - now)
                now = loop.time()
            self._next_request_time = max(self._next_request_time, now) + interval
        return

    def _retry_delay(self, retry_idx: int) -> float:
        delay = self._retry_min_delay * (2**retry_idx)
        return min(delay, self._retry_max_delay)

    async def _run_with_retry(self, client: RemoteEncoderBase, inputs):
        for attempt in range(self._retry_times + 1):
            await self._wait_for_rpm()
            try:
                return await client.async_encode(inputs)
            except Exception:
                if attempt >= self._retry_times:
                    raise
                delay = self._retry_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)
        raise RuntimeError("Remote encoder retry loop exited unexpectedly.")

    async def _async_encode_core(
        self,
        inputs: list[ContentPart],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        batches = [
            inputs[i : i + self._batch_size]
            for i in range(0, len(inputs), self._batch_size)
        ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(inputs), interval=log_interval, display=display
        ) as p_logger:

            async def _encode_task(idx: int, batch: list[ContentPart]) -> None:
                async with semaphore:
                    res = await self._run_with_retry(
                        client,
                        self._prepare_encoder_inputs(batch),
                    )
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

        return _merge_encode_results(results)


class ProcessEncoderAdapter(EncoderRuntimeAdapter):
    """Process-backed runtime adapter for local raw encoders.

    Local raw encoders process canonical batches synchronously. This adapter
    creates a process worker pool lazily, splits managed calls into deployment
    batches, dispatches batches across available workers, and merges results
    back into input order.
    """

    impl_cls: type[LocalEncoderBase] | None = None

    def __init__(
        self,
        config,
        impl_cls: type[LocalEncoderBase] | None = None,
        *,
        input_format: EncoderInputFormat | None = None,
        batch_size: int = 32,
        device_groups: list[list[int]] | None = None,
    ):
        """Create a process-backed encoder runtime adapter.

        :param config: Configuration passed to the raw encoder implementation.
        :param impl_cls: Optional raw encoder implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param input_format: Canonical input format expected by the raw encoder.
            ``"text"`` extracts text strings from text content blocks.
        :param batch_size: Deployment batch size used for worker RPC calls.
        :param device_groups: Worker device placement. ``None`` creates one
            worker inheriting the current environment, ``[]`` creates one
            CPU-only worker, and non-empty groups create one worker per group.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        super().__init__(config, input_format=input_format)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self._batch_size = batch_size
        self._device_groups = device_groups
        self._worker_count = 1
        self._embedding_size = None
        return

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        client = ProcessWorkerPoolClient.from_device_groups(
            self.impl_cls,
            config,
            self._device_groups,
        )
        self._worker_count = len(client)
        self._embedding_size = await client.call_primary("embedding_size")
        return client

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return self._worker_count

    async def _async_encode_core(
        self,
        inputs: list[ContentPart],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        batches = [
            inputs[i : i + self._batch_size]
            for i in range(0, len(inputs), self._batch_size)
        ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | np.ndarray] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(inputs), interval=log_interval, display=display
        ) as p_logger:

            async def _encode_task(idx: int, batch: list[ContentPart]) -> None:
                async with semaphore:
                    res = await client.call_available(
                        "_encode_batch",
                        self._prepare_encoder_inputs(batch),
                    )
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

        return _merge_encode_results(results)

    async def _async_call_primary(self, attribute: str, *args, **kwargs):
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        async with semaphore:
            return await client.call_primary(attribute, *args, **kwargs)

    def _call_primary(self, attribute: str, *args, **kwargs):
        return self._run_coroutine_sync(
            self._async_call_primary(attribute, *args, **kwargs)
        )

    @property
    def embedding_size(self):
        if self._embedding_size is None:
            self._embedding_size = self._call_primary("embedding_size")
        return self._embedding_size
