from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from flexrag.common import Context
from flexrag.retrievers import (
    FlexRetriever,
    FlexRetrieverConfig,
    Hit,
    LMDBContextStore,
    LMDBContextStoreConfig,
    RetrievalView,
    SyncCollectionBackendBase,
)
from flexrag.retrievers.backends.base import AsyncCollectionBackendBase


def contexts(total: int = 5) -> Iterable[Context]:
    for idx in range(total):
        yield Context(context_id=f"doc-{idx}", data={"text": f"text {idx}"})


class RecordingBackend(SyncCollectionBackendBase):
    requires_context_store = False

    def __init__(self, *, addable: bool, fail_once: bool = False) -> None:
        super().__init__(RetrievalView("view", ["text"]))
        self._is_addable = addable
        self.fail_once = fail_once
        self.add_batch_sizes: list[int] = []
        self.rebuild_count = 0
        self.clear_count = 0
        self.context_ids: list[str] = []

    @property
    def is_addable(self) -> bool:
        return self._is_addable

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        items = list(contexts)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("intentional failure")
        if not self.is_addable:
            super().add_contexts(items)
        self.add_batch_sizes.append(len(items))
        self.context_ids.extend(ctx.context_id for ctx in items)

    def rebuild(self, contexts: Iterable[Context]) -> None:
        self.rebuild_count += 1
        self.context_ids = [ctx.context_id for ctx in contexts]

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        hits = [
            Hit(context_id=context_id, score=1.0, backend="", view="view")
            for context_id in self.context_ids[:top_k]
        ]
        return [list(hits) for _ in queries]

    def clear(self) -> None:
        self.clear_count += 1
        self.context_ids = []

    def count(self) -> int:
        return len(set(self.context_ids))


class AsyncNativeBackend(AsyncCollectionBackendBase):
    requires_context_store = False
    is_addable = True

    def __init__(self) -> None:
        super().__init__(RetrievalView("async", ["text"]))

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        return

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        return

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        return [[Hit("doc-async", 1.0, "", "async")] for _ in queries]

    async def async_clear(self) -> None:
        return

    async def async_count(self) -> int:
        return 1


def make_retriever(
    root: Path,
) -> tuple[FlexRetriever, RecordingBackend, RecordingBackend]:
    store = LMDBContextStore(LMDBContextStoreConfig(path=root / "contexts"))
    addable = RecordingBackend(addable=True)
    rebuild = RecordingBackend(addable=False)
    return (
        FlexRetriever.from_backends(
            {"addable": addable, "rebuild": rebuild},
            context_store=store,
            config=FlexRetrieverConfig(batch_size=2),
        ),
        addable,
        rebuild,
    )


def test_add_contexts_updates_addable_and_rebuild_backends(tmp_path: Path) -> None:
    retriever, addable, rebuild = make_retriever(tmp_path)
    retriever.add_contexts(contexts())
    assert addable.add_batch_sizes == [2, 2, 1]
    assert rebuild.rebuild_count == 1
    assert retriever.count() == 5
    assert retriever.search("q", top_k=1, used_backends=["rebuild"])[0][0].data == {
        "text": "text 0"
    }
    assert retriever.context_store is not None
    retriever.context_store.close()


def test_add_and_remove_backend(tmp_path: Path) -> None:
    retriever, _, _ = make_retriever(tmp_path)
    retriever.add_contexts(contexts())
    extra = RecordingBackend(addable=False)
    retriever.add_backend("extra", extra)
    assert extra.rebuild_count == 1
    with pytest.raises(ValueError):
        retriever.add_backend("extra", extra)
    assert retriever.remove_backend("extra", clear=True) is extra
    assert extra.clear_count == 1
    assert retriever.context_store is not None
    retriever.context_store.close()


@pytest.mark.asyncio
async def test_async_api_matches_sync_semantics(tmp_path: Path) -> None:
    retriever, addable, rebuild = make_retriever(tmp_path)
    await retriever.async_add_contexts(contexts())
    assert addable.add_batch_sizes == [2, 2, 1]
    assert rebuild.rebuild_count == 1
    assert await retriever.async_count() == 5
    assert (await retriever.async_search("q", top_k=1))[0][0].context_id == "doc-0"
    await retriever.async_clear()
    assert await retriever.async_count() == 0
    assert retriever.context_store is not None
    await retriever.context_store.async_close()


@pytest.mark.asyncio
async def test_async_native_backend_sync_bridge_fails_in_loop() -> None:
    with pytest.raises(RuntimeError, match="async_\\*"):
        AsyncNativeBackend().search_hits(["q"], 1)


def test_rebuild_recovers_after_add_failure(tmp_path: Path) -> None:
    store = LMDBContextStore(LMDBContextStoreConfig(path=tmp_path / "recovery"))
    failing = RecordingBackend(addable=True, fail_once=True)
    retriever = FlexRetriever.from_backends({"failing": failing}, context_store=store)
    with pytest.raises(RuntimeError):
        retriever.add_contexts(contexts(3))
    assert store.count() == 3
    assert failing.count() == 0
    retriever.rebuild("failing")
    assert failing.count() == 3
    store.close()


def test_count_uses_backend_counts_without_context_store() -> None:
    retriever = FlexRetriever.from_backends(
        {"left": RecordingBackend(addable=True), "right": RecordingBackend(addable=True)}
    )
    retriever.backends["left"].context_ids = ["a", "b"]
    retriever.backends["right"].context_ids = ["c", "d"]
    assert retriever.count() == 2
    retriever.backends["right"].context_ids = ["c"]
    with pytest.raises(RuntimeError):
        retriever.count()


def test_non_addable_backend_without_store_fails_before_consuming_input() -> None:
    consumed = False

    def generator() -> Iterable[Context]:
        nonlocal consumed
        consumed = True
        yield from contexts(1)

    retriever = FlexRetriever.from_backends({"rebuild": RecordingBackend(addable=False)})
    with pytest.raises(ValueError):
        retriever.add_contexts(generator())
    assert consumed is False
