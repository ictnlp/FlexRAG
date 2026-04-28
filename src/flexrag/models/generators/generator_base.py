import asyncio
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Optional, Protocol

from flexrag.common import ChatMessages, ChatTurn, Register, configure
from flexrag.common.logging import SimpleProgressLogger
from flexrag.models.async_client_base import AsyncClientMixin, ConfigT


@configure
class GenerationConfig:
    """Configuration for text generation.
    Note that not all options are supported by all models.

    :param do_sample: Whether to use sampling for generation. Defaults to True.
    :type do_sample: bool
    :param sample_num: The number of samples to generate. Defaults to 1.
    :type sample_num: int
    :param temperature: The temperature of the sampling distribution. Defaults to 1.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of tokens to generate. Defaults to None.
        None means no limit.
    :type max_new_tokens: Optional[int]
    :param top_p: The cumulative probability for nucleus sampling. Defaults to None.
    :type top_p: Optional[float]
    :param top_k: The number of tokens to consider for top-k sampling. Defaults to None.
    :type top_k: Optional[int]
    :param eos_token_id: The token id for the end of sentence token. Defaults to None.
    :type eos_token_id: Optional[int]
    :param stop_str: A list of strings to stop generation. Defaults to [].
    :type stop_str: list[str]
    :param tools: Provider-native tool definitions passed through to supported chat models.
        Defaults to [].
    :type tools: list[dict[str, Any]]
    :param reasoning_effort: Provider-specific reasoning effort hint. Defaults to None.
    :type reasoning_effort: Optional[str]
    :param response_format: OpenAI compatible schema constraint.
        Defaults to None.
    :type response_format: Optional[dict[str, Any]]
    """

    do_sample: bool = True
    sample_num: int = 1
    temperature: float = 1.0
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_str: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    reasoning_effort: Optional[str] = None
    response_format: Optional[dict[str, Any]] = None

    def __post_init__(self):
        assert self.sample_num > 0, "sample_num must be greater than 0"
        if self.sample_num > 1:
            assert self.do_sample, "do_sample must be True when sample_num > 1"
        assert self.temperature >= 0, "temperature must be greater than or equal to 0"
        if self.max_new_tokens is not None:
            assert self.max_new_tokens > 0, "max_new_tokens must be greater than 0"
        if self.top_p is not None:
            assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1"
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be greater than 0"


class GeneratorProtocol(Protocol):
    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[ChatTurn]]: ...

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[ChatTurn]]: ...

    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[str]]: ...

    async def async_generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
        batch_size: int = 1,
        log_interval: int = 1000,
    ) -> list[list[str]]: ...


class GeneratorBase(AsyncClientMixin[ConfigT], ABC):
    """Base class for client-backed generator implementations."""

    def __init__(self, config: ConfigT):
        AsyncClientMixin.__init__(self, config)
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
    ) -> list[list[ChatTurn]]:
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
    ) -> list[list[ChatTurn]]:
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
        results: list[None | list[list[ChatTurn]]] = [None] * len(batches)

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
        merged: list[list[ChatTurn]] = []
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
    ) -> list[list[ChatTurn]]:
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
    ) -> list[list[ChatTurn]]:
        return self._run_coroutine_sync(
            self._async_chat_core(
                messages,
                generation_config=generation_config,
                batch_size=batch_size,
                log_interval=log_interval,
            )
        )


GENERATORS = Register[GeneratorProtocol]("generator")
