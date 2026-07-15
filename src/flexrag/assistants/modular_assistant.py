from __future__ import annotations

import asyncio
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import field
from typing import cast

from flexrag.common import ChatMessages, ChatTurn, Context, RetrievedContext, configure
from flexrag.models.generators import GenerationConfig, GeneratorProtocol
from flexrag.processors.rankers import RankerProtocol
from flexrag.processors.refiners import RefinerProtocol
from flexrag.retrievers import RetrieverProtocol

from .assistant_base import ASSISTANTS, AssistantBase, AssistantResult


@configure
class ModularAssistantConfig:
    """Configuration for the modular assistant.

    :param generation_config: Options forwarded to the injected generator.
    :param used_fields: Context data fields rendered into the generation prompt.
        An empty list renders every field.
    """

    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    used_fields: list[str] = field(default_factory=list)


@ASSISTANTS("modular", config_class=ModularAssistantConfig)
class ModularAssistant(AssistantBase):
    """Reference implementation of a modular retrieval-augmented assistant.

    ``ModularAssistant`` represents the conventional Modular RAG architecture:
    generation, static knowledge retrieval, episode memory retrieval, ranking,
    and context refinement are independent components composed into a fixed
    response pipeline. Concrete models and retrieval backends are supplied by
    the caller, allowing the same structure to support plain QA, contextual QA,
    RAG, and retrieval-based memory evaluation.

    Compared with simple retrieve-then-generate pipelines, it supports richer
    composition and post-processing while retaining a predictable execution
    flow. Unlike an agentic assistant, it does not select tools or repeatedly
    plan and act.

    Example:

    .. code-block:: python

        from flexrag.assistants import ModularAssistant, ModularAssistantConfig

        async def answer_question(generator, retriever):
            assistant = ModularAssistant(
                ModularAssistantConfig(),
                generator=generator,
                retriever=retriever,
            )

            async with assistant:
                result = await assistant.answer(
                    [
                        {
                            "role": "user",
                            "content": "What is retrieval-augmented generation?",
                        }
                    ]
                )

            return result.response.text_content
    """

    def __init__(
        self,
        config: ModularAssistantConfig,
        *,
        generator: GeneratorProtocol,
        retriever: RetrieverProtocol | None = None,
        memory_retriever: RetrieverProtocol | None = None,
        ranker: RankerProtocol | None = None,
        refiners: Sequence[RefinerProtocol] = (),
    ) -> None:
        """Compose an assistant from injected, non-owned components.

        :param config: Generation and context-rendering configuration.
        :param generator: Asynchronous chat generator.
        :param retriever: Optional read-only static knowledge retriever.
        :param memory_retriever: Optional writable episode memory retriever.
        :param ranker: Optional asynchronous candidate ranker.
        :param refiners: Asynchronous refiners applied in order.
        :raises ValueError: If static knowledge and episode memory use the same
            retriever object.
        """
        super().__init__()
        if retriever is not None and retriever is memory_retriever:
            raise ValueError(
                "retriever and memory_retriever must be different components"
            )
        self.config = config
        self.generator = generator
        self.retriever = retriever
        self.memory_retriever = memory_retriever
        self.ranker = ranker
        self.refiners = tuple(refiners)
        self._memory_populated = False
        self._memory_sequence = 0

    async def _start_episode(self) -> None:
        self._memory_populated = False
        self._memory_sequence = 0
        if self.memory_retriever is not None:
            await self.memory_retriever.async_clear()

    async def _finish_episode(self) -> None:
        try:
            if self.memory_retriever is not None:
                await self.memory_retriever.async_clear()
        finally:
            self._memory_populated = False

    async def _add_histories(self, histories: Sequence[ChatMessages]) -> None:
        memory = self._require_memory()
        contexts = [self._history_context(history) for history in histories]
        if contexts:
            await memory.async_add_contexts(contexts)
            self._memory_populated = True

    async def _add_contexts(self, contexts: Sequence[Context]) -> None:
        memory = self._require_memory()
        normalized = [self._memory_context(context) for context in contexts]
        if normalized:
            await memory.async_add_contexts(normalized)
            self._memory_populated = True

    async def _answer(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool = True,
    ) -> AssistantResult:
        return await self._respond(messages, retrieve=retrieve)

    def _ensure_run_supported(self) -> None:
        super()._ensure_run_supported()
        self._require_memory()

    async def _commit(
        self,
        messages: ChatMessages,
        result: AssistantResult,
    ) -> None:
        memory = self._require_memory()
        await memory.async_add_contexts([self._turn_context(messages, result.response)])
        self._memory_populated = True

    async def _respond(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool,
    ) -> AssistantResult:
        search_memory = self.memory_retriever is not None and self._memory_populated
        search_knowledge = retrieve and self.retriever is not None
        query: str | None = None
        if search_memory or search_knowledge:
            query = self._last_user_text(messages)

        memory_contexts: list[RetrievedContext] = []
        knowledge_contexts: list[RetrievedContext] = []
        searches: list[tuple[str, asyncio.Task[list[list[RetrievedContext]]]]] = []
        if search_memory:
            assert self.memory_retriever is not None and query is not None
            searches.append(
                (
                    "memory",
                    asyncio.create_task(self.memory_retriever.async_search([query])),
                )
            )
        if search_knowledge:
            assert self.retriever is not None and query is not None
            searches.append(
                (
                    "knowledge",
                    asyncio.create_task(self.retriever.async_search([query])),
                )
            )
        if searches:
            results = await asyncio.gather(*(task for _, task in searches))
            for (source, _), batches in zip(searches, results, strict=True):
                contexts = self._single_search_result(source, batches)
                if source == "memory":
                    memory_contexts = contexts
                else:
                    knowledge_contexts = contexts

        candidates = self._deduplicate([*memory_contexts, *knowledge_contexts])
        if self.ranker is not None and candidates:
            if query is None:
                query = self._last_user_text(messages)
            ranking = await self.ranker.async_rank(query, candidates)
            candidates = self._validate_contexts(ranking.candidates, "ranker")

        if candidates and self.refiners:
            if query is None:
                query = self._optional_last_user_text(messages)
            if query is not None:
                for context in candidates:
                    if context.query is None:
                        context.query = query
        for refiner in self.refiners:
            if not candidates:
                break
            candidates = self._validate_contexts(
                await refiner.async_refine(candidates), "refiner"
            )

        prompt = self._build_prompt(messages, candidates)
        responses = await self.generator.async_chat(
            prompt,
            generation_config=self.config.generation_config,
        )
        return AssistantResult(
            response=self._first_response(responses),
            contexts=candidates,
            trajectory=[],
            metadata={
                "prompt": prompt,
                "context_counts": {
                    "memory": len(memory_contexts),
                    "knowledge": len(knowledge_contexts),
                    "final": len(candidates),
                },
            },
        )

    def _history_context(self, history: ChatMessages) -> Context:
        metadata = deepcopy(history.metadata)
        metadata["_assistant_memory_kind"] = "history"
        return Context(
            context_id=self._next_memory_id("history"),
            data={"text": self._render_messages(history)},
            source="history",
            metadata=metadata,
        )

    def _memory_context(self, context: Context) -> Context:
        metadata = deepcopy(context.metadata)
        metadata["original_context_id"] = context.context_id
        metadata["_assistant_memory_kind"] = "context"
        return Context(
            context_id=self._next_memory_id("context"),
            data=deepcopy(context.data),
            source=context.source,
            metadata=metadata,
        )

    def _turn_context(self, messages: ChatMessages, response: ChatTurn) -> Context:
        transcript = self._render_messages(messages)
        response_text = response.text_content or "[non-text content]"
        return Context(
            context_id=self._next_memory_id("turn"),
            data={"text": f"{transcript}\nassistant: {response_text}"},
            source="assistant",
            metadata={"_assistant_memory_kind": "turn"},
        )

    def _next_memory_id(self, prefix: str) -> str:
        self._memory_sequence += 1
        return f"{prefix}-{self._memory_sequence}"

    @staticmethod
    def _render_messages(messages: ChatMessages) -> str:
        return "\n".join(
            f"{turn.role}: {turn.text_content or '[non-text content]'}"
            for turn in messages
        )

    @classmethod
    def _single_search_result(
        cls,
        source: str,
        batches: list[list[RetrievedContext]],
    ) -> list[RetrievedContext]:
        if len(batches) != 1:
            raise RuntimeError(
                f"{source} retriever returned {len(batches)} result batches for one query"
            )
        return cls._validate_contexts(deepcopy(batches[0]), f"{source} retriever")

    @staticmethod
    def _validate_contexts(
        contexts: Sequence[RetrievedContext | str],
        component: str,
    ) -> list[RetrievedContext]:
        if not all(isinstance(context, RetrievedContext) for context in contexts):
            raise TypeError(f"{component} must return RetrievedContext candidates")
        return cast(list[RetrievedContext], list(contexts))

    @staticmethod
    def _deduplicate(
        contexts: Sequence[RetrievedContext],
    ) -> list[RetrievedContext]:
        unique: list[RetrievedContext] = []
        seen_ids: set[str] = set()
        for context in contexts:
            if context.context_id:
                if context.context_id in seen_ids:
                    continue
                seen_ids.add(context.context_id)
            unique.append(context)
        return unique

    def _build_prompt(
        self,
        messages: ChatMessages,
        contexts: Sequence[RetrievedContext],
    ) -> ChatMessages:
        prompt = deepcopy(messages)
        if not contexts:
            return prompt

        user_index = self._last_user_index(prompt)
        rendered = "\n\n".join(
            f"Context {index}: {self._render_context(context)}"
            for index, context in enumerate(contexts, start=1)
        )
        prefix = (
            "Here are some context documents that may be relevant to this "
            f"conversation:\n\n{rendered}\n\n"
        )
        content = prompt[user_index].content
        if isinstance(content, str):
            prompt[user_index].content = f"{prefix}{content}"
        else:
            prompt[user_index].content = [{"type": "text", "text": prefix}, *content]
        return prompt

    def _render_context(self, context: RetrievedContext) -> str:
        fields = self.config.used_fields
        if not fields:
            return "\n".join(
                f"{field_name}: {field_value}"
                for field_name, field_value in context.data.items()
            )
        if len(fields) == 1:
            return str(context.data[fields[0]])
        return "\n".join(
            f"{field_name}: {context.data[field_name]}" for field_name in fields
        )

    @classmethod
    def _last_user_text(cls, messages: ChatMessages) -> str:
        text = messages[cls._last_user_index(messages)].text_content
        if not text:
            raise ValueError("retrieval and ranking require text in the last user turn")
        return text

    @classmethod
    def _optional_last_user_text(cls, messages: ChatMessages) -> str | None:
        try:
            turn = messages[cls._last_user_index(messages)]
        except ValueError:
            return None
        return turn.text_content or None

    @staticmethod
    def _last_user_index(messages: ChatMessages) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                return index
        raise ValueError("context injection and retrieval require a user turn")

    @staticmethod
    def _first_response(responses: list[list[ChatTurn]]) -> ChatTurn:
        if not responses or not responses[0]:
            raise RuntimeError("generator returned no response candidates")
        response = responses[0][0]
        if not isinstance(response, ChatTurn):
            raise TypeError("generator must return ChatTurn candidates")
        return response

    def _require_memory(self) -> RetrieverProtocol:
        if self.memory_retriever is None:
            raise RuntimeError("this operation requires a memory_retriever")
        return self.memory_retriever
