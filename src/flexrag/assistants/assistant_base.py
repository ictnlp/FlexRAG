from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import field
from types import TracebackType
from typing import Any, Protocol, Self

from flexrag.common import Register, data
from flexrag.common.dataclasses import ChatMessages, ChatTurn, Context, RetrievedContext


@data
class AssistantResult:
    """Result of one assistant execution.

    :param response: Assistant response turn.
    :param contexts: Final contexts used to produce the response.
    :param trajectory: Intermediate assistant turns, such as agent steps.
    :param metadata: Execution metadata and implementation-specific evidence.
    """

    response: ChatTurn
    contexts: list[RetrievedContext] = field(default_factory=list)
    trajectory: list[ChatTurn] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class AssistantProtocol(Protocol):
    """Asynchronous interface for an episode-scoped evaluation assistant."""

    async def __aenter__(self) -> Self:
        """Start a fresh evaluation episode.

        :return: The assistant initialized for the episode.
        :raises RuntimeError: If the assistant is already inside an episode.
        """
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish the current episode.

        Injected components are not closed by this operation.

        :param exc_type: Type of an exception raised in the context, if any.
        :param exc: Exception raised in the context, if any.
        :param traceback: Exception traceback, if any.
        """
        ...

    async def add_histories(self, histories: Sequence[ChatMessages]) -> None:
        """Persist completed conversations in the current episode.

        :param histories: Conversation histories with optional session metadata.
        :raises RuntimeError: If no episode is active or history persistence is
            unsupported.
        """
        ...

    async def add_contexts(self, contexts: Sequence[Context]) -> None:
        """Persist contexts in the current episode.

        :param contexts: Contexts to copy and add to episode state.
        :raises RuntimeError: If no episode is active or context persistence is
            unsupported.
        """
        ...

    async def answer(
        self,
        messages: ChatMessages | list[dict[str, Any]],
        *,
        retrieve: bool = True,
    ) -> AssistantResult:
        """Answer without modifying episode state.

        :param messages: Conversation to answer.
        :param retrieve: Whether implementation-specific static retrieval is
            enabled.
        :return: The response and execution evidence.
        :raises RuntimeError: If no episode is active.
        """
        ...

    async def run(
        self,
        messages: ChatMessages | list[dict[str, Any]],
        *,
        retrieve: bool = True,
    ) -> AssistantResult:
        """Answer and commit implementation-defined episode state.

        :param messages: Conversation to execute.
        :param retrieve: Whether implementation-specific static retrieval is
            enabled.
        :return: The response and execution evidence.
        :raises RuntimeError: If no episode is active or stateful execution is
            unsupported.
        """
        ...


class _EpisodeGate:
    """Writer-preferring async gate for episode reads and mutations."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    @asynccontextmanager
    async def read(self) -> AsyncIterator[None]:
        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._writer and self._waiting_writers == 0
            )
            self._readers += 1
        try:
            yield
        finally:
            async with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @asynccontextmanager
    async def write(self) -> AsyncIterator[None]:
        async with self._condition:
            self._waiting_writers += 1
            try:
                await self._condition.wait_for(
                    lambda: not self._writer and self._readers == 0
                )
            finally:
                self._waiting_writers -= 1
                self._condition.notify_all()
            self._writer = True
        try:
            yield
        finally:
            async with self._condition:
                self._writer = False
                self._condition.notify_all()


class AssistantBase(ABC):
    """Base class for asynchronous assistants evaluated by FlexRAG tasks.

    An assistant receives a conversation and returns an
    :class:`AssistantResult`. Use it inside an ``async with`` block, where each
    block represents one independent evaluation episode.

    Every subclass must implement ``_answer()``. This method represents a
    read-only interaction: it may be called concurrently and must not modify
    episode state.

    Additional methods depend on the tasks the assistant supports:

    - QA, RAG, and contextual QA assistants only need ``_answer()``.
    - Assistants for static memory evaluation should also implement
      ``_add_histories()`` and/or ``_add_contexts()`` to initialize episode
      memory before questions are answered.
    - Assistants for dynamic memory tasks may implement ``_commit()`` to record
      each completed turn. Stateful ``run()`` support is inferred from this
      method.
    - Agentic assistants may override ``_run()`` when execution and state
      updates form a custom agent loop.
    - Override ``_start_episode()`` and ``_finish_episode()`` when per-episode
      state or temporary clients need to be initialized and reset.

    Optional capabilities that are not implemented raise ``RuntimeError``.
    Components supplied to an assistant remain owned by the caller.
    """

    def __init__(self) -> None:
        self._active = False
        self._gate = _EpisodeGate()
        self._lifecycle_lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Start a fresh episode through the subclass lifecycle hook.

        :return: This assistant, ready for episode operations.
        :raises RuntimeError: If the assistant is already inside an episode.
        """
        async with self._lifecycle_lock:
            if self._active:
                raise RuntimeError("Assistant is already inside an episode")
            await self._start_episode()
            self._active = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish the episode after active operations have completed.

        Injected components remain owned by the caller.

        :param exc_type: Type of an exception raised in the context, if any.
        :param exc: Exception raised in the context, if any.
        :param traceback: Exception traceback, if any.
        """
        del exc_type, exc, traceback
        async with self._lifecycle_lock:
            if not self._active:
                return
            self._active = False
            async with self._gate.write():
                await self._finish_episode()

    async def add_histories(self, histories: Sequence[ChatMessages]) -> None:
        """Persist histories while excluding reads and other writes.

        :param histories: Histories copied before reaching the subclass hook.
        :raises RuntimeError: If no episode is active or history persistence is
            unsupported.
        """
        self._require_active()
        copied = deepcopy(tuple(histories))
        async with self._gate.write():
            self._require_active()
            await self._add_histories(copied)

    async def add_contexts(self, contexts: Sequence[Context]) -> None:
        """Persist contexts while excluding reads and other writes.

        :param contexts: Contexts copied before reaching the subclass hook.
        :raises RuntimeError: If no episode is active or context persistence is
            unsupported.
        """
        self._require_active()
        copied = deepcopy(tuple(contexts))
        async with self._gate.write():
            self._require_active()
            await self._add_contexts(copied)

    async def answer(
        self,
        messages: ChatMessages | list[dict[str, Any]],
        *,
        retrieve: bool = True,
    ) -> AssistantResult:
        """Answer through the read-only subclass hook.

        :param messages: Conversation copied and normalized before execution.
        :param retrieve: Implementation-specific static retrieval switch.
        :return: The response and execution evidence.
        :raises RuntimeError: If no episode is active.
        """
        self._require_active()
        normalized = self._normalize_messages(messages)
        async with self._gate.read():
            self._require_active()
            return await self._answer(normalized, retrieve=retrieve)

    async def run(
        self,
        messages: ChatMessages | list[dict[str, Any]],
        *,
        retrieve: bool = True,
    ) -> AssistantResult:
        """Execute a stateful turn under exclusive episode access.

        :param messages: Conversation copied and normalized before execution.
        :param retrieve: Implementation-specific static retrieval switch.
        :return: The response and execution evidence.
        :raises RuntimeError: If no episode is active or stateful execution is
            unsupported.
        """
        self._require_active()
        self._ensure_run_supported()
        normalized = self._normalize_messages(messages)
        async with self._gate.write():
            self._require_active()
            self._ensure_run_supported()
            return await self._run(normalized, retrieve=retrieve)

    async def _start_episode(self) -> None:
        return

    async def _finish_episode(self) -> None:
        return

    async def _add_histories(self, histories: Sequence[ChatMessages]) -> None:
        del histories
        raise RuntimeError("this assistant does not support episode memory")

    async def _add_contexts(self, contexts: Sequence[Context]) -> None:
        del contexts
        raise RuntimeError("this assistant does not support episode memory")

    @abstractmethod
    async def _answer(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool,
    ) -> AssistantResult:
        raise NotImplementedError

    async def _run(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool,
    ) -> AssistantResult:
        result = await self._answer(messages, retrieve=retrieve)
        await self._commit(messages, result)
        return result

    def _ensure_run_supported(self) -> None:
        """Validate that the subclass implements a stateful execution path."""
        cls = type(self)
        if cls._run is AssistantBase._run and cls._commit is AssistantBase._commit:
            raise RuntimeError("this assistant does not support stateful run")

    async def _commit(
        self,
        messages: ChatMessages,
        result: AssistantResult,
    ) -> None:
        del messages, result
        raise RuntimeError("this assistant does not support stateful run")

    @staticmethod
    def _normalize_messages(
        messages: ChatMessages | list[dict[str, Any]],
    ) -> ChatMessages:
        if isinstance(messages, ChatMessages):
            return deepcopy(messages)
        return ChatMessages.from_list(deepcopy(messages))

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Assistant methods require an active episode")


ASSISTANTS = Register[AssistantProtocol]("assistant")
