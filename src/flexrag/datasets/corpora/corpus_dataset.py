from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Protocol, overload, runtime_checkable

from flexrag.common import Context, Register


@runtime_checkable
class IterableCorpus(Protocol):
    """Protocol for corpora that can stream :class:`Context` objects."""

    def __iter__(self) -> Iterator[Context]:
        """Iterate over contexts in the corpus."""
        ...


@runtime_checkable
class MappingCorpus(IterableCorpus, Protocol):
    """Protocol for materialized corpora with positional and id-based access."""

    @overload
    def __getitem__(self, index: int) -> Context:
        """Get the context at the given position."""
        ...

    @overload
    def __getitem__(self, index: slice) -> "CorpusView":
        """Get a view over a slice of the corpus."""
        ...

    def __getitem__(self, index: int | slice) -> Context | "CorpusView":
        """Get the context at the given position or a sliced corpus view."""
        ...

    @property
    def contexts(self) -> Mapping[str, Context]:
        """Return a mapping of context ids to :class:`Context` objects."""
        ...

    @property
    def context_ids(self) -> Iterator[str]:
        """Iterate over context ids in the corpus."""
        ...

    def __len__(self) -> int:
        """Return the number of contexts in the corpus."""
        ...


CORPORA = Register[IterableCorpus]("corpus")


class _InMemoryMappingCorpus:
    """Shared helpers for corpora that materialize contexts in memory."""

    _contexts: dict[str, Context] | None = None
    _ordered_contexts: tuple[Context, ...] | None = None

    def _require_materialized_contexts(self, caller: str) -> dict[str, Context]:
        if self._contexts is None:
            raise RuntimeError(
                f"{self.__class__.__name__}.{caller} requires load_in_memory=True."
            )
        return self._contexts

    def _set_materialized_contexts(
        self, contexts: Mapping[str, Context] | None
    ) -> None:
        if contexts is None:
            self._contexts = None
            self._ordered_contexts = None
            return
        self._contexts = dict(contexts)
        self._ordered_contexts = tuple(self._contexts.values())
        return

    @property
    def contexts(self) -> Mapping[str, Context]:
        return MappingProxyType(self._require_materialized_contexts("contexts"))

    def __len__(self) -> int:
        return len(self._require_materialized_contexts("__len__"))

    @overload
    def __getitem__(self, index: int) -> Context: ...

    @overload
    def __getitem__(self, index: slice) -> "CorpusView": ...

    def __getitem__(self, index: int | slice) -> Context | "CorpusView":
        ordered_contexts = self._ordered_contexts
        if ordered_contexts is None:
            self._require_materialized_contexts("__getitem__")
            ordered_contexts = self._ordered_contexts
        assert ordered_contexts is not None
        if isinstance(index, slice):
            return CorpusView(self, tuple(range(len(ordered_contexts))[index]))
        if isinstance(index, int):
            return ordered_contexts[index]
        raise TypeError(
            f"Corpus indices must be integers or slices, not {type(index).__name__}."
        )


class CorpusView:
    """Read-only view over a subset of a materialized corpus."""

    def __init__(self, corpus: MappingCorpus, indices: list[int] | tuple[int, ...]):
        self._corpus = corpus
        self._indices = tuple(indices)
        self._contexts: Mapping[str, Context] | None = None
        return

    def __iter__(self) -> Iterator[Context]:
        for index in self._indices:
            yield self._corpus[index]
        return

    def __len__(self) -> int:
        return len(self._indices)

    @overload
    def __getitem__(self, index: int) -> Context: ...

    @overload
    def __getitem__(self, index: slice) -> "CorpusView": ...

    def __getitem__(self, index: int | slice) -> Context | "CorpusView":
        if isinstance(index, slice):
            return CorpusView(self._corpus, self._indices[index])
        if isinstance(index, int):
            return self._corpus[self._indices[index]]
        raise TypeError(
            f"Corpus indices must be integers or slices, not {type(index).__name__}."
        )

    @property
    def contexts(self) -> Mapping[str, Context]:
        if self._contexts is None:
            self._contexts = MappingProxyType(
                {context.context_id: context for context in self}
            )
        return self._contexts

    @property
    def context_ids(self) -> Iterator[str]:
        yield from self.contexts.keys()
        return


class _ContextMappingCorpus(_InMemoryMappingCorpus):
    """Internal mapping-backed corpus view for dataset-owned context stores."""

    def __init__(self, contexts: Mapping[str, Context]):
        self._set_materialized_contexts(contexts)
        return

    def __iter__(self) -> Iterator[Context]:
        assert self._ordered_contexts is not None
        yield from self._ordered_contexts
        return

    @property
    def context_ids(self) -> Iterator[str]:
        yield from self.contexts.keys()
        return


__all__ = ["CORPORA", "CorpusView", "IterableCorpus", "MappingCorpus"]
