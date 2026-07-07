from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from flexrag.common import Context
from flexrag.retrievers import (
    LMDBContextStore,
    LMDBContextStoreConfig,
    SQLiteContextStore,
    SQLiteContextStoreConfig,
)


def contexts() -> list[Context]:
    return [
        Context(context_id="doc-a", data={"text": "alpha"}, meta_data={"rank": 1}),
        Context(context_id="doc-b", data={"text": "beta"}, meta_data={"rank": 2}),
        Context(context_id="doc-c", data={"text": "gamma"}, meta_data={"rank": 3}),
    ]


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda path: LMDBContextStore(LMDBContextStoreConfig(path=path / "lmdb")),
        lambda path: SQLiteContextStore(
            SQLiteContextStoreConfig(path=path / "sqlite.db")
        ),
    ],
)
def test_context_store_sync_contract(
    tmp_path: Path,
    store_factory: Callable[[Path], Any],
) -> None:
    store = store_factory(tmp_path)
    store.set_many(contexts())
    assert store.count() == 3
    assert store.ids == ["doc-a", "doc-b", "doc-c"]
    assert store.get("doc-b").data["text"] == "beta"
    assert [ctx.context_id for ctx in store.get_many(["doc-c", "doc-a"])] == [
        "doc-c",
        "doc-a",
    ]
    assert [ctx.context_id for ctx in store.iter_contexts()] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]
    with pytest.raises(KeyError):
        store.get("missing")
    store.close()

    reopened = store_factory(tmp_path)
    assert reopened.count() == 3
    assert reopened.get("doc-a").meta_data["rank"] == 1
    reopened.clear()
    assert reopened.count() == 0
    reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store_factory",
    [
        lambda path: LMDBContextStore(LMDBContextStoreConfig(path=path / "lmdb")),
        lambda path: SQLiteContextStore(
            SQLiteContextStoreConfig(path=path / "sqlite.db")
        ),
    ],
)
async def test_context_store_async_bridge(
    tmp_path: Path,
    store_factory: Callable[[Path], Any],
) -> None:
    store = store_factory(tmp_path)
    await store.async_set_many(contexts())
    assert await store.async_count() == 3
    assert await store.async_ids() == ["doc-a", "doc-b", "doc-c"]
    assert (await store.async_get("doc-b")).data["text"] == "beta"
    assert [
        ctx.context_id for ctx in await store.async_get_many(["doc-c", "doc-a"])
    ] == ["doc-c", "doc-a"]
    assert [ctx.context_id async for ctx in store.async_iter_contexts()] == [
        "doc-a",
        "doc-b",
        "doc-c",
    ]
    await store.async_clear()
    assert await store.async_count() == 0
    await store.async_close()
