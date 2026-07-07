from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from typing import Protocol

from flexrag.common.dataclasses import Context


class ContextStoreProtocol(Protocol):
    """Structural contract for complete ``Context`` storage.

    Context stores own the full context payload and provide the corpus source
    used for hydration, counting, and full backend rebuilds.
    """

    def set_many(self, contexts: Iterable[Context]) -> None:
        """Store or replace multiple contexts.

        :param contexts: Context objects to persist.
        """
        ...

    async def async_set_many(self, contexts: Iterable[Context]) -> None:
        """Asynchronously store or replace multiple contexts.

        :param contexts: Context objects to persist.
        """
        ...

    def get(self, context_id: str) -> Context:
        """Return a context by id.

        :param context_id: Context identifier to fetch.
        :returns: The stored context.
        :raises KeyError: If the id is missing.
        """
        ...

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously return a context by id.

        :param context_id: Context identifier to fetch.
        :returns: The stored context.
        :raises KeyError: If the id is missing.
        """
        ...

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Return contexts for the requested ids.

        :param context_ids: Context identifiers to fetch.
        :returns: Contexts in the same order as ``context_ids``.
        :raises KeyError: If any id is missing.
        """
        ...

    async def async_get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Asynchronously return contexts for the requested ids.

        :param context_ids: Context identifiers to fetch.
        :returns: Contexts in the same order as ``context_ids``.
        :raises KeyError: If any id is missing.
        """
        ...

    def iter_contexts(self) -> Iterable[Context]:
        """Iterate over all stored contexts.

        :returns: Iterable over complete stored contexts.
        """
        ...

    def async_iter_contexts(self) -> AsyncIterator[Context]:
        """Asynchronously iterate over all stored contexts.

        :returns: Async iterator over complete stored contexts.
        """
        ...

    @property
    def ids(self) -> list[str]:
        """Return all stored context ids."""
        ...

    async def async_ids(self) -> list[str]:
        """Asynchronously return all stored context ids."""
        ...

    def count(self) -> int:
        """Return the number of stored contexts."""
        ...

    async def async_count(self) -> int:
        """Asynchronously return the number of stored contexts."""
        ...

    def clear(self) -> None:
        """Delete all stored contexts without necessarily deleting artifacts."""
        ...

    async def async_clear(self) -> None:
        """Asynchronously delete all stored contexts."""
        ...

    def close(self) -> None:
        """Release store resources."""
        ...

    async def async_close(self) -> None:
        """Asynchronously release store resources."""
        ...


class SyncContextStoreBase(ABC):
    """Base class for sync-native context stores.

    Subclasses implement synchronous storage methods. Async methods are
    thread-backed compatibility bridges and are not native async streaming.
    """

    @abstractmethod
    def set_many(self, contexts: Iterable[Context]) -> None:
        """Store or replace multiple contexts.

        :param contexts: Context objects to persist.
        """
        raise NotImplementedError

    async def async_set_many(self, contexts: Iterable[Context]) -> None:
        """Asynchronously store contexts through ``set_many``.

        :param contexts: Context objects to persist.
        """
        await asyncio.to_thread(self.set_many, contexts)
        return

    @abstractmethod
    def get(self, context_id: str) -> Context:
        """Return a context by id.

        :param context_id: Context identifier to fetch.
        :returns: The stored context.
        :raises KeyError: If the id is missing.
        """
        raise NotImplementedError

    async def async_get(self, context_id: str) -> Context:
        """Asynchronously return a context through ``get``.

        :param context_id: Context identifier to fetch.
        :returns: The stored context.
        :raises KeyError: If the id is missing.
        """
        return await asyncio.to_thread(self.get, context_id)

    @abstractmethod
    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Return contexts for the requested ids.

        :param context_ids: Context identifiers to fetch.
        :returns: Contexts in the same order as ``context_ids``.
        :raises KeyError: If any id is missing.
        """
        raise NotImplementedError

    async def async_get_many(self, context_ids: Iterable[str]) -> list[Context]:
        """Asynchronously return contexts through ``get_many``.

        :param context_ids: Context identifiers to fetch.
        :returns: Contexts in the same order as ``context_ids``.
        :raises KeyError: If any id is missing.
        """
        return await asyncio.to_thread(self.get_many, context_ids)

    @abstractmethod
    def iter_contexts(self) -> Iterable[Context]:
        """Iterate over all stored contexts.

        :returns: Iterable over complete stored contexts.
        """
        raise NotImplementedError

    async def async_iter_contexts(self) -> AsyncIterator[Context]:
        """Asynchronously iterate over all contexts via a thread bridge.

        The sync iterator is materialized in a worker thread before yielding.

        :returns: Async iterator over complete stored contexts.
        """
        contexts = await asyncio.to_thread(lambda: list(self.iter_contexts()))
        for context in contexts:
            yield context
        return

    @property
    @abstractmethod
    def ids(self) -> list[str]:
        """Return all stored context ids."""
        raise NotImplementedError

    async def async_ids(self) -> list[str]:
        """Asynchronously return all stored context ids."""
        return await asyncio.to_thread(lambda: self.ids)

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored contexts."""
        raise NotImplementedError

    async def async_count(self) -> int:
        """Asynchronously return the number of stored contexts."""
        return await asyncio.to_thread(self.count)

    @abstractmethod
    def clear(self) -> None:
        """Delete all stored contexts without necessarily deleting artifacts."""
        raise NotImplementedError

    async def async_clear(self) -> None:
        """Asynchronously delete all stored contexts through ``clear``."""
        await asyncio.to_thread(self.clear)
        return

    def close(self) -> None:
        """Release store resources.

        The default implementation is a no-op for stores without explicit
        resources.
        """
        return

    async def async_close(self) -> None:
        """Asynchronously release store resources through ``close``."""
        await asyncio.to_thread(self.close)
        return
