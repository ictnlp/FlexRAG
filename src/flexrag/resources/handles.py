from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from flexrag.common import ChatTurn, ProgressDisplay
    from flexrag.common.dataclasses import Context, RetrievedContext
    from flexrag.models.encoders.encoder_base import EncoderInputs
    from flexrag.models.generators.generator_base import (
        GenerationConfig,
        GeneratorMessages,
        GeneratorPrefixes,
    )
    from flexrag.models.scorers.scorer_base import PairScorerInput
    from flexrag.processors.chunkers import Chunk
    from flexrag.processors.rankers.ranker_base import RankingResult


class RuntimeHandleBase:
    """Base class for managed runtime handles."""

    required_methods: tuple[str, ...] = ()
    required_attributes: tuple[str, ...] = ()

    def __init__(self, resource: Any):
        self._resource = resource
        self._validate_resource()
        return

    def _validate_resource(self) -> None:
        missing = [
            name
            for name in self.required_methods
            if not callable(getattr(self._resource, name, None))
        ]
        missing.extend(
            name
            for name in self.required_attributes
            if not hasattr(self._resource, name)
        )
        if missing:
            missing_items = ", ".join(missing)
            raise TypeError(
                f"{self.__class__.__name__} requires resource methods or "
                f"attributes: {missing_items}."
            )
        return


class EncoderHandle(RuntimeHandleBase):
    """Managed encoder runtime handle."""

    required_methods = ("encode", "async_encode")

    def encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return self._resource.encode(
            inputs,
            log_interval=log_interval,
            display=display,
        )

    async def async_encode(
        self,
        inputs: EncoderInputs,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return await self._resource.async_encode(
            inputs,
            log_interval=log_interval,
            display=display,
        )

    @property
    def embedding_size(self) -> int | None:
        try:
            return getattr(self._resource, "embedding_size", None)
        except AttributeError:
            return None


class GeneratorHandle(RuntimeHandleBase):
    """Managed generator runtime handle."""

    required_methods = ("chat", "async_chat", "generate", "async_generate")

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        return self._resource.chat(
            messages,
            generation_config=generation_config,
            log_interval=log_interval,
            display=display,
        )

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[ChatTurn]]:
        return await self._resource.async_chat(
            messages,
            generation_config=generation_config,
            log_interval=log_interval,
            display=display,
        )

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        return self._resource.generate(
            prefixes,
            generation_config=generation_config,
            log_interval=log_interval,
            display=display,
        )

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> list[list[str]]:
        return await self._resource.async_generate(
            prefixes,
            generation_config=generation_config,
            log_interval=log_interval,
            display=display,
        )


class ScorerHandle(RuntimeHandleBase):
    """Managed pair scorer runtime handle."""

    required_methods = ("score", "async_score")

    def score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return self._resource.score(
            pairs,
            log_interval=log_interval,
            display=display,
        )

    async def async_score(
        self,
        pairs: PairScorerInput,
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> np.ndarray:
        return await self._resource.async_score(
            pairs,
            log_interval=log_interval,
            display=display,
        )


class RankerHandle(RuntimeHandleBase):
    """Managed ranker runtime handle."""

    required_methods = ("rank", "async_rank")

    def rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        return self._resource.rank(
            query,
            candidates,
            log_interval=log_interval,
            display=display,
        )

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
        log_interval: int = 1000,
        display: ProgressDisplay = "auto",
    ) -> RankingResult:
        return await self._resource.async_rank(
            query,
            candidates,
            log_interval=log_interval,
            display=display,
        )


class RefinerHandle(RuntimeHandleBase):
    """Managed refiner runtime handle."""

    required_methods = ("refine",)

    def refine(
        self,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        return self._resource.refine(contexts)


class ChunkerHandle(RuntimeHandleBase):
    """Managed chunker runtime handle."""

    required_methods = ("chunk",)

    def chunk(self, text: str, return_str: bool = False) -> list[Chunk] | list[str]:
        """Split text into chunks.

        :param text: Text to split.
        :param return_str: Whether to return chunk strings instead of chunk
            objects.
        :return: Chunk objects or chunk strings.
        """
        return self._resource.chunk(text, return_str=return_str)


class ContextStoreHandle(RuntimeHandleBase):
    """Managed context-store runtime handle."""

    required_methods = (
        "set_many",
        "async_set_many",
        "get",
        "async_get",
        "get_many",
        "async_get_many",
        "iter_contexts",
        "async_iter_contexts",
        "async_ids",
        "count",
        "async_count",
        "clear",
        "async_clear",
    )
    required_attributes = ("ids",)

    def set_many(self, contexts: Iterable[Context]) -> None:
        """Store or replace multiple contexts."""
        self._resource.set_many(contexts)
        return

    async def async_set_many(self, contexts: Iterable[Context]) -> None:
        """Asynchronously store or replace multiple contexts."""
        await self._resource.async_set_many(contexts)
        return

    def get(self, context_id: str) -> Context:
        """Return a context by id."""
        return self._resource.get(context_id)

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously return a context by id."""
        return await self._resource.async_get(context_id)

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Return contexts for the requested ids."""
        return self._resource.get_many(context_ids)

    async def async_get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Asynchronously return contexts for the requested ids."""
        return await self._resource.async_get_many(context_ids)

    def iter_contexts(self) -> Iterable[Context]:
        """Iterate over all stored contexts."""
        return self._resource.iter_contexts()

    async def async_iter_contexts(self) -> AsyncIterator[Context]:
        """Asynchronously iterate over all stored contexts."""
        async for context in self._resource.async_iter_contexts():
            yield context
        return

    @property
    def ids(self) -> list[str]:
        """Return all stored context ids."""
        return self._resource.ids

    async def async_ids(self) -> list[str]:
        """Asynchronously return all stored context ids."""
        return await self._resource.async_ids()

    def count(self) -> int:
        """Return the number of stored contexts."""
        return self._resource.count()

    async def async_count(self) -> int:
        """Asynchronously return the number of stored contexts."""
        return await self._resource.async_count()

    def clear(self) -> None:
        """Delete all stored contexts without deleting artifacts."""
        self._resource.clear()
        return

    async def async_clear(self) -> None:
        """Asynchronously delete all stored contexts."""
        await self._resource.async_clear()
        return


class TokenizerHandle(RuntimeHandleBase):
    """Managed tokenizer runtime handle."""

    required_methods = (
        "tokenize",
        "detokenize",
        "encode",
        "decode",
        "tokens_to_ids",
        "ids_to_tokens",
    )
    required_attributes = ("reversible", "vocab_size")

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into string tokens."""
        return self._resource.tokenize(text)

    def detokenize(self, tokens: list[str]) -> str:
        """Convert string tokens back to text."""
        return self._resource.detokenize(tokens)

    def encode(self, text: str) -> list[int]:
        """Encode text into token ids."""
        return self._resource.encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode token ids into text."""
        return self._resource.decode(tokens)

    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert string tokens to token ids."""
        return self._resource.tokens_to_ids(tokens)

    def ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        """Convert token ids to string tokens."""
        return self._resource.ids_to_tokens(token_ids)

    @property
    def reversible(self) -> bool:
        """Return whether tokenization is strictly reversible."""
        return self._resource.reversible

    @property
    def vocab_size(self) -> int:
        """Return the tokenizer vocabulary size."""
        return self._resource.vocab_size
