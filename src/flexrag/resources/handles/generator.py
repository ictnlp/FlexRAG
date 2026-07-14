from __future__ import annotations

from flexrag.common import ChatTurn
from flexrag.models.generators.generator_base import (
    GenerationConfig,
    GeneratorMessages,
    GeneratorPrefixes,
    _normalize_chat_messages,
    _normalize_generation_prefixes,
)

from ..runtime import RuntimeCall
from .base import TypedHandle


class GeneratorHandle(TypedHandle):
    """Typed proxy for generator resources.

    The handle normalizes generation and chat inputs, batches public calls when
    supported, submits primitive runtime calls, and flattens per-batch results.
    It does not own the generator lifecycle.
    """

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> list[list[str]]:
        """Synchronously generate completions for prefixes.

        :param prefixes: Prefix input accepted by the formal generator API.
        :param generation_config: Optional generation settings for this call.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Generated sample lists in input order.
        """
        normalized = _normalize_generation_prefixes(prefixes)
        if not normalized:
            return []

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "generate",
                args=(batch,),
                kwargs={
                    "generation_config": generation_config,
                    "batch_size": batch_size,
                },
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results: list[list[list[str]]] = self._target.batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Generating",
        )
        return [item for batch_result in results for item in batch_result]

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> list[list[str]]:
        """Asynchronously generate completions for prefixes.

        :param prefixes: Prefix input accepted by the formal generator API.
        :param generation_config: Optional generation settings for this call.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Generated sample lists in input order.
        """
        normalized = _normalize_generation_prefixes(prefixes)
        if not normalized:
            return []

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "async_generate",
                args=(batch,),
                kwargs={
                    "generation_config": generation_config,
                    "batch_size": batch_size,
                },
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results = await self._target.async_batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Generating",
        )
        return [item for batch_result in results for item in batch_result]

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> list[list[ChatTurn]]:
        """Synchronously generate chat responses.

        :param messages: Chat messages accepted by the formal generator API.
        :param generation_config: Optional generation settings for this call.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Generated chat turn lists in input order.
        """
        normalized = _normalize_chat_messages(messages)
        if not normalized:
            return []

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "chat",
                args=(batch,),
                kwargs={
                    "generation_config": generation_config,
                    "batch_size": batch_size,
                },
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results: list[list[list[ChatTurn]]] = self._target.batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Chatting",
        )
        return [item for batch_result in results for item in batch_result]

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: str = "auto",
    ) -> list[list[ChatTurn]]:
        """Asynchronously generate chat responses.

        :param messages: Chat messages accepted by the formal generator API.
        :param generation_config: Optional generation settings for this call.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :returns: Generated chat turn lists in input order.
        """
        normalized = _normalize_chat_messages(messages)
        if not normalized:
            return []

        batch_size = self._effective_batch_size
        calls = [
            RuntimeCall(
                "async_chat",
                args=(batch,),
                kwargs={
                    "generation_config": generation_config,
                    "batch_size": batch_size,
                },
                weight=len(batch),
            )
            for batch in self._batches(normalized)
        ]
        results: list[list[list[ChatTurn]]] = await self._target.async_batch_call(
            calls,
            log_interval=log_interval,
            display=display,
            desc="Chatting",
        )
        return [item for batch_result in results for item in batch_result]
