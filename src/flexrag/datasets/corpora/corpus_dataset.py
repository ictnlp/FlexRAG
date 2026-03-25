from __future__ import annotations

from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from flexrag.common import Context, Register


@runtime_checkable
class IterableCorpus(Protocol):
    """Protocol for corpora that can stream :class:`Context` objects."""

    def __iter__(self) -> Iterator[Context]:
        """Iterate over contexts in the corpus."""
        ...


@runtime_checkable
class MappingCorpus(IterableCorpus, Protocol):
    """Protocol for corpora that expose id-addressable contexts and length."""

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


class _ContextMappingCorpus:
    """Internal mapping-backed corpus view for dataset-owned context stores."""

    def __init__(self, contexts: Mapping[str, Context]):
        self._contexts = dict(contexts)
        return

    def __iter__(self) -> Iterator[Context]:
        yield from self._contexts.values()
        return

    def __len__(self) -> int:
        return len(self._contexts)

    @property
    def contexts(self) -> Mapping[str, Context]:
        return MappingProxyType(self._contexts)

    @property
    def context_ids(self) -> Iterator[str]:
        yield from self._contexts.keys()
        return


__all__ = ["CORPORA", "IterableCorpus", "MappingCorpus"]
