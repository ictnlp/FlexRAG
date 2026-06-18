import asyncio
from abc import abstractmethod

from flexrag.common import ChatMessages, ChatTurn, ProgressDisplay, SimpleProgressLogger
from flexrag.models.generators.generator_base import (
    GenerationConfig,
    LocalGeneratorBase,
    RemoteGeneratorBase,
)
from flexrag.runtime.async_client import AsyncClientMixin, ConfigT
from flexrag.runtime.process_worker_pool import ProcessWorkerPoolClient


def _normalize_chat_messages(
    messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
) -> list[ChatMessages]:
    if isinstance(messages, ChatMessages):
        return [messages]
    if not messages:
        return []
    if isinstance(messages[0], dict):
        return [ChatMessages.from_list(messages)]

    normalized: list[ChatMessages] = []
    for message in messages:
        if isinstance(message, ChatMessages):
            normalized.append(message)
        else:
            normalized.append(ChatMessages.from_list(message))
    return normalized


class GeneratorRuntimeAdapter(AsyncClientMixin[ConfigT]):
    """Managed call adapter for generator-like resources.

    This base adapter provides the public sync and async call surface for
    generator resources. It normalizes accepted input shapes, submits work to
    the managed background event loop, and delegates the actual generation work
    to subclass core methods.
    """

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @abstractmethod
    async def _async_generate_core(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        return

    @abstractmethod
    async def _async_chat_core(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        return

    async def async_generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions asynchronously.

        ``prefixes`` may be a single string or a list of strings. The adapter
        normalizes it to a batch and returns one list of candidate completions
        per prefix, preserving input order.

        :param prefixes: Prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Batched candidate completions.
        """
        normalized_prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
        return await self._run_coroutine_async(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions synchronously.

        ``prefixes`` may be a single string or a list of strings. The adapter
        normalizes it to a batch and runs the async generation core on the
        managed background event loop.

        :param prefixes: Prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Batched candidate completions.
        """
        normalized_prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
        return self._run_coroutine_sync(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously.

        ``messages`` may be one ``ChatMessages`` object, one list of message
        dictionaries, a batch of ``ChatMessages`` objects, or a batch of message
        dictionary lists. The adapter normalizes it to ``list[ChatMessages]``
        before dispatch.

        :param messages: Conversation or conversations to continue.
        :param generation_config: Optional generation options for this call.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Batched candidate assistant turns.
        """
        normalized_messages = _normalize_chat_messages(messages)
        return await self._run_coroutine_async(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses synchronously.

        ``messages`` may be one ``ChatMessages`` object, one list of message
        dictionaries, a batch of ``ChatMessages`` objects, or a batch of message
        dictionary lists. The adapter normalizes it to ``list[ChatMessages]``
        and runs the async chat core on the managed background event loop.

        :param messages: Conversation or conversations to continue.
        :param generation_config: Optional generation options for this call.
        :param log_interval: Progress update interval.
        :param display: Progress display mode.
        :return: Batched candidate assistant turns.
        """
        normalized_messages = _normalize_chat_messages(messages)
        return self._run_coroutine_sync(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )


class RemoteGeneratorRuntimeAdapter(GeneratorRuntimeAdapter[ConfigT]):
    """Managed runtime adapter for remote raw generators.

    Remote raw generators expose single-sample asynchronous core methods. This
    adapter fans batched managed calls out to those cores while preserving
    result order. It also owns remote-runtime policies such as maximum
    concurrency, requests-per-minute limiting, retry delays, and progress
    logging.
    """

    impl_cls: type[RemoteGeneratorBase] | None = None

    def __init__(
        self,
        config: ConfigT,
        impl_cls: type[RemoteGeneratorBase] | None = None,
        *,
        max_concurrency: int = 1,
        rpm: float = 0,
        retry_times: int = 0,
        retry_min_delay: float = 1.0,
        retry_max_delay: float = 60.0,
    ) -> None:
        """Create a remote generator runtime adapter.

        :param config: Configuration passed to the raw generator implementation.
        :param impl_cls: Optional raw generator implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param max_concurrency: Maximum number of in-flight single-sample
            remote requests.
        :param rpm: Requests-per-minute limit. ``0`` disables rate limiting.
        :param retry_times: Number of retries after the initial attempt. ``0``
            disables retry.
        :param retry_min_delay: Initial retry delay in seconds.
        :param retry_max_delay: Maximum retry delay in seconds.
        :raises ValueError: If any runtime policy value is invalid.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
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

    async def _run_with_retry(self, call, *args):
        for attempt in range(self._retry_times + 1):
            await self._wait_for_rpm()
            try:
                return await call(*args)
            except Exception:
                if attempt >= self._retry_times:
                    raise
                delay = self._retry_delay(attempt)
            if delay > 0:
                await asyncio.sleep(delay)
        raise RuntimeError("Remote generator retry loop exited unexpectedly.")

    async def _async_generate_core(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | list[str]] = [None] * len(prefixes)

        with SimpleProgressLogger(
            total=len(prefixes), interval=log_interval, display=display
        ) as p_logger:

            async def _generate_task(idx: int, prefix: str) -> None:
                async with semaphore:
                    res = await self._run_with_retry(
                        client._async_generate_one,
                        prefix,
                        generation_config,
                    )
                results[idx] = res
                p_logger.update(1, desc="Generating")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, prefix in enumerate(prefixes):
                        tg.create_task(
                            _generate_task(idx, prefix), name=f"generate_{idx}"
                        )
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some generate tasks did not produce results.")
        return ready_results

    async def _async_chat_core(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | list[ChatTurn]] = [None] * len(messages)

        with SimpleProgressLogger(
            total=len(messages), interval=log_interval, display=display
        ) as p_logger:

            async def _chat_task(idx: int, message: ChatMessages) -> None:
                async with semaphore:
                    res = await self._run_with_retry(
                        client._async_chat_one,
                        message,
                        generation_config,
                    )
                results[idx] = res
                p_logger.update(1, desc="Chatting")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, message in enumerate(messages):
                        tg.create_task(_chat_task(idx, message), name=f"chat_{idx}")
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some chat tasks did not produce results.")
        return ready_results


class ProcessGeneratorAdapter(GeneratorRuntimeAdapter):
    """Process-backed runtime adapter for local raw generators.

    Local raw generators process canonical batches synchronously. This adapter
    creates a process worker pool lazily, splits managed calls into deployment
    batches, dispatches batches across available workers, and merges results
    back into input order.
    """

    impl_cls: type[LocalGeneratorBase] | None = None

    def __init__(
        self,
        config,
        impl_cls: type[LocalGeneratorBase] | None = None,
        *,
        batch_size: int = 1,
    ):
        """Create a process-backed generator runtime adapter.

        :param config: Configuration passed to the raw generator implementation.
        :param impl_cls: Optional raw generator implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param batch_size: Deployment batch size used for worker RPC calls.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        super().__init__(config)
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self._batch_size = batch_size
        self._worker_count = 1
        return

    def _build_worker_device_groups(self, config) -> list[list[int] | None]:
        device_ids = list(getattr(config, "device_id", []))
        if not device_ids:
            return [None]
        if getattr(config, "parallel_mode", None) == "pipeline":
            return [device_ids]
        return [[device_id] for device_id in device_ids]

    async def _create_client(self, config):
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        client = ProcessWorkerPoolClient.from_worker_groups(
            self.impl_cls,
            config,
            self._build_worker_device_groups(config),
        )
        self._worker_count = len(client)
        return client

    async def _close_client(self, client) -> None:
        await client.close()
        return

    def _get_max_concurrency(self) -> int:
        return self._worker_count

    async def _async_generate_core(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        batches = [
            prefixes[i : i + self._batch_size]
            for i in range(0, len(prefixes), self._batch_size)
        ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | list[list[str]]] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(prefixes), interval=log_interval, display=display
        ) as p_logger:

            async def _generate_task(idx: int, batch: list[str]) -> None:
                async with semaphore:
                    res = await client.call_available(
                        "_generate_batch",
                        batch,
                        generation_config=generation_config,
                    )
                results[idx] = res
                p_logger.update(len(batch), desc="Generating")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _generate_task(idx, batch), name=f"generate_batch_{idx}"
                        )
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some generate tasks did not produce results.")
        merged: list[list[str]] = []
        for batch_result in ready_results:
            merged.extend(batch_result)
        return merged

    async def _async_chat_core(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        batches = [
            messages[i : i + self._batch_size]
            for i in range(0, len(messages), self._batch_size)
        ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        results: list[None | list[list[ChatTurn]]] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(messages), interval=log_interval, display=display
        ) as p_logger:

            async def _chat_task(idx: int, batch: list[ChatMessages]) -> None:
                async with semaphore:
                    res = await client.call_available(
                        "_chat_batch",
                        batch,
                        generation_config=generation_config,
                    )
                results[idx] = res
                p_logger.update(len(batch), desc="Chatting")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(_chat_task(idx, batch), name=f"chat_batch_{idx}")
            except ExceptionGroup as exc:
                raise self._unwrap_exception_group(exc) from exc

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some chat tasks did not produce results.")
        merged: list[list[ChatTurn]] = []
        for batch_result in ready_results:
            merged.extend(batch_result)
        return merged
