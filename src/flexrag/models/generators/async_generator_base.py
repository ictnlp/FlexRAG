import asyncio
from abc import abstractmethod

from flexrag.common.dataclasses import ChatMessages
from flexrag.common.logging import SimpleProgressLogger
from flexrag.models.async_client_base import AsyncClientMixin, ConfigT

from .generator_base import GenerationConfig, GeneratorBase


class AsyncGeneratorBase(GeneratorBase, AsyncClientMixin[ConfigT]):
    """Base class for generator proxies backed by an async client/runtime."""

    def __init__(self, config: ConfigT):
        AsyncClientMixin.__init__(self, config)
        return

    def close(self) -> None:
        AsyncClientMixin.close(self)
        return

    @staticmethod
    def _unwrap_exception_group(exc: Exception):
        while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        return exc

    @staticmethod
    def _normalize_messages(
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

    @abstractmethod
    async def _async_generate_impl(
        self,
        client,
        prefixes: list[str],
        generation_config: GenerationConfig | None,
    ) -> list[list[str]]:
        return

    @abstractmethod
    async def _async_chat_impl(
        self,
        client,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None,
    ) -> list[list]:
        return

    async def _async_generate_core(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[str]]:
        prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
        if batch_size is None:
            batches = [prefixes]
        else:
            batches = [
                prefixes[i : i + batch_size]
                for i in range(0, len(prefixes), batch_size)
            ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        p_logger = SimpleProgressLogger(total=len(prefixes), interval=log_interval)
        results: list[None | list[list[str]]] = [None] * len(batches)

        async def _generate_task(idx: int, batch: list[str]) -> None:
            async with semaphore:
                res = await self._async_generate_impl(client, batch, generation_config)
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
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ):
        normalized_messages = self._normalize_messages(messages)
        if batch_size is None:
            batches = [normalized_messages]
        else:
            batches = [
                normalized_messages[i : i + batch_size]
                for i in range(0, len(normalized_messages), batch_size)
            ]

        client = await self._get_async_client()
        semaphore = await self._get_async_semaphore()
        p_logger = SimpleProgressLogger(
            total=len(normalized_messages), interval=log_interval
        )
        results: list[None | list[list]] = [None] * len(batches)

        async def _chat_task(idx: int, batch: list[ChatMessages]) -> None:
            async with semaphore:
                res = await self._async_chat_impl(client, batch, generation_config)
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
        merged: list[list] = []
        for batch_result in ready_results:
            merged.extend(batch_result)
        return merged

    async def async_generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[str]]:
        return await self._run_coroutine_async(
            self._async_generate_core(
                prefixes,
                generation_config=generation_config,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[str]]:
        return self._run_coroutine_sync(
            self._async_generate_core(
                prefixes,
                generation_config=generation_config,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ):
        return await self._run_coroutine_async(
            self._async_chat_core(
                messages,
                generation_config=generation_config,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )

    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ):
        return self._run_coroutine_sync(
            self._async_chat_core(
                messages,
                generation_config=generation_config,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )
