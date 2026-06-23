import asyncio
from typing import Any, TypeVar

from flexrag.common import ChatMessages, ChatTurn, ProgressDisplay, SimpleProgressLogger
from flexrag.models.generators.generator_base import (
    GenerationConfig,
    GeneratorMessages,
    GeneratorPrefixes,
    _normalize_chat_messages,
    _normalize_generation_prefixes,
)

from .common import split_batches, unwrap_exception_group

T = TypeVar("T")


def _collect_ready_results(results: list[None | T], desc: str) -> list[T]:
    ready_results = [result for result in results if result is not None]
    if len(ready_results) != len(results):
        raise RuntimeError(f"Some {desc} tasks did not produce results.")
    return ready_results


class BatchGeneratorInvocation:
    """Invocation semantics for batch-native managed generators."""

    def __init__(
        self,
        runtime: Any,
        *,
        generate_method: str,
        chat_method: str,
        batch_size: int = 1,
    ) -> None:
        """Create a batch generator invocation.

        :param runtime: Runtime adapter used to execute primitive calls.
        :param generate_method: Primitive method for one prefix batch.
        :param chat_method: Primitive method for one chat batch.
        :param batch_size: Deployment batch size.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.runtime = runtime
        self._generate_method = generate_method
        self._chat_method = chat_method
        self._batch_size = batch_size
        return

    async def _async_generate_core(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        batches = split_batches(prefixes, self._batch_size)
        results: list[None | list[list[str]]] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(prefixes), interval=log_interval, display=display
        ) as p_logger:

            async def _generate_task(idx: int, batch: list[str]) -> None:
                results[idx] = await self.runtime.acall(
                    self._generate_method,
                    batch,
                    generation_config=generation_config,
                )
                p_logger.update(len(batch), desc="Generating")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(
                            _generate_task(idx, batch),
                            name=f"generate_batch_{idx}",
                        )
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        merged: list[list[str]] = []
        for batch_result in _collect_ready_results(results, "generate"):
            merged.extend(batch_result)
        return merged

    async def _async_chat_core(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        batches = split_batches(messages, self._batch_size)
        results: list[None | list[list[ChatTurn]]] = [None] * len(batches)

        with SimpleProgressLogger(
            total=len(messages), interval=log_interval, display=display
        ) as p_logger:

            async def _chat_task(idx: int, batch: list[ChatMessages]) -> None:
                results[idx] = await self.runtime.acall(
                    self._chat_method,
                    batch,
                    generation_config=generation_config,
                )
                p_logger.update(len(batch), desc="Chatting")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, batch in enumerate(batches):
                        tg.create_task(_chat_task(idx, batch), name=f"chat_batch_{idx}")
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        merged: list[list[ChatTurn]] = []
        for batch_result in _collect_ready_results(results, "chat"):
            merged.extend(batch_result)
        return merged

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions asynchronously."""
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        return await self.runtime.run_async(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions synchronously."""
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        return self.runtime.run_sync(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously."""
        normalized_messages = _normalize_chat_messages(messages)
        return await self.runtime.run_async(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses synchronously."""
        normalized_messages = _normalize_chat_messages(messages)
        return self.runtime.run_sync(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )


class SingleSampleGeneratorInvocation:
    """Invocation semantics for single-sample remote managed generators."""

    def __init__(
        self,
        runtime: Any,
        *,
        generate_method: str,
        chat_method: str,
    ) -> None:
        """Create a single-sample generator invocation.

        :param runtime: Runtime adapter used to execute primitive calls.
        :param generate_method: Primitive method for one prefix.
        :param chat_method: Primitive method for one chat conversation.
        """
        self.runtime = runtime
        self._generate_method = generate_method
        self._chat_method = chat_method
        return

    async def _async_generate_core(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        results: list[None | list[str]] = [None] * len(prefixes)

        with SimpleProgressLogger(
            total=len(prefixes), interval=log_interval, display=display
        ) as p_logger:

            async def _generate_task(idx: int, prefix: str) -> None:
                results[idx] = await self.runtime.acall(
                    self._generate_method,
                    prefix,
                    generation_config,
                )
                p_logger.update(1, desc="Generating")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, prefix in enumerate(prefixes):
                        tg.create_task(
                            _generate_task(idx, prefix),
                            name=f"generate_{idx}",
                        )
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        return _collect_ready_results(results, "generate")

    async def _async_chat_core(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        results: list[None | list[ChatTurn]] = [None] * len(messages)

        with SimpleProgressLogger(
            total=len(messages), interval=log_interval, display=display
        ) as p_logger:

            async def _chat_task(idx: int, message: ChatMessages) -> None:
                results[idx] = await self.runtime.acall(
                    self._chat_method,
                    message,
                    generation_config,
                )
                p_logger.update(1, desc="Chatting")
                return

            try:
                async with asyncio.TaskGroup() as tg:
                    for idx, message in enumerate(messages):
                        tg.create_task(_chat_task(idx, message), name=f"chat_{idx}")
            except ExceptionGroup as exc:
                raise unwrap_exception_group(exc) from exc

        return _collect_ready_results(results, "chat")

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions asynchronously."""
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        return await self.runtime.run_async(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        """Generate completions synchronously."""
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        return self.runtime.run_sync(
            self._async_generate_core(
                normalized_prefixes,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously."""
        normalized_messages = _normalize_chat_messages(messages)
        return await self.runtime.run_async(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                log_interval=log_interval,
                display=display,
            )
        )

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        """Generate chat responses synchronously."""
        normalized_messages = _normalize_chat_messages(messages)
        return self.runtime.run_sync(
            self._async_chat_core(
                normalized_messages,
                generation_config=generation_config,
                display=display,
                log_interval=log_interval,
            )
        )
