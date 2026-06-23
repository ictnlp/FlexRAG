from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

import numpy as np

if TYPE_CHECKING:
    from flexrag.common import ChatTurn, ProgressDisplay
    from flexrag.common.dataclasses import RetrievedContext
    from flexrag.models.encoders.encoder_base import EncoderInputs
    from flexrag.models.generators.generator_base import (
        GenerationConfig,
        GeneratorMessages,
        GeneratorPrefixes,
    )
    from flexrag.models.scorers.scorer_base import PairScorerInput
    from flexrag.processors.rankers.ranker_base import RankingResult

DEFAULT_INDEX_BATCH_SIZE = 512


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
    ) -> RankingResult:
        return self._resource.rank(query, candidates)

    async def async_rank(
        self,
        query: str,
        candidates: list[RetrievedContext | str],
    ) -> RankingResult:
        return await self._resource.async_rank(query, candidates)


class RefinerHandle(RuntimeHandleBase):
    """Managed refiner runtime handle."""

    required_methods = ("refine",)

    def refine(
        self,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        return self._resource.refine(contexts)


class IndexHandle(RuntimeHandleBase):
    """Managed retriever index runtime handle."""

    required_methods = ("build_index", "insert", "search", "save_to_local", "clear")
    required_attributes = ("cfg", "is_addable", "infimum", "supremum")

    @property
    def cfg(self) -> Any:
        return self._resource.cfg

    @property
    def is_addable(self) -> bool:
        return self._resource.is_addable

    @property
    def infimum(self) -> float:
        return self._resource.infimum

    @property
    def supremum(self) -> float:
        return self._resource.supremum

    def build_index(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        scratch_path: str | None = None,
    ) -> None:
        return self._resource.build_index(
            context_ids,
            data,
            batch_size=batch_size,
            scratch_path=scratch_path,
        )

    def insert_batch(
        self,
        context_ids: Iterable[str],
        data: Iterable[dict[str, Any]],
        batch_size: int = DEFAULT_INDEX_BATCH_SIZE,
        log_interval: int = 10000,
        display: str = "auto",
    ) -> None:
        return self._resource.insert_batch(
            context_ids,
            data,
            batch_size=batch_size,
            log_interval=log_interval,
            display=display,
        )

    def insert(self, context_ids: list[str], data: list[dict[str, Any]]) -> None:
        return self._resource.insert(context_ids, data)

    def search(
        self,
        query: list[Any],
        top_k: int,
        **search_kwargs,
    ) -> tuple[list[list[str]], np.ndarray]:
        return self._resource.search(query, top_k, **search_kwargs)

    def save_to_local(self, index_path: str) -> None:
        return self._resource.save_to_local(index_path)

    def clear(self) -> None:
        return self._resource.clear()

    def __len__(self) -> int:
        return len(self._resource)
