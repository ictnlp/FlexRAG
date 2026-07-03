import time

from flexrag.common.runtime_cache import (
    MemoryRuntimeCache,
    NullRuntimeCache,
    RuntimeCacheConfig,
    SQLiteRuntimeCache,
    make_runtime_cache_key,
)


def test_runtime_cache_key_is_stable():
    assert make_runtime_cache_key({"b": 2, "a": [1, 2]}) == make_runtime_cache_key(
        {"a": [1, 2], "b": 2}
    )


def test_memory_runtime_cache_get_set_clear_and_prune():
    cache = MemoryRuntimeCache("test", RuntimeCacheConfig(max_entries=2))
    cache.set_many({"a": {"value": 1}, "b": {"value": 2}})
    assert cache.get_many(["a", "b", "c"]) == [{"value": 1}, {"value": 2}, None]

    time.sleep(0.001)
    assert cache.get_many(["a"]) == [{"value": 1}]
    cache.set_many({"c": {"value": 3}})
    assert cache.get_many(["a", "b", "c"]) == [{"value": 1}, None, {"value": 3}]

    cache.clear()
    assert cache.get_many(["a", "c"]) == [None, None]


def test_memory_runtime_cache_ttl():
    cache = MemoryRuntimeCache("test", RuntimeCacheConfig(ttl_seconds=0.001))
    cache.set_many({"a": {"value": 1}})
    assert cache.get_many(["a"]) == [{"value": 1}]
    time.sleep(0.01)
    assert cache.get_many(["a"]) == [None]


def test_sqlite_runtime_cache_is_lazy_and_persistent(tmp_path):
    config = RuntimeCacheConfig(mode="disk", cache_dir=tmp_path)
    cache = SQLiteRuntimeCache("test", config)
    assert not (tmp_path / "runtime_cache.sqlite3").exists()

    cache.set_many({"a": {"value": 1}}, metadata={"source": "test"})
    assert (tmp_path / "runtime_cache.sqlite3").exists()
    assert cache.get_many(["a", "b"]) == [{"value": 1}, None]
    exported = list(cache.items())
    assert exported[0]["key"] == "a"
    assert exported[0]["value"] == {"value": 1}
    assert exported[0]["metadata"] == {"source": "test"}
    cache.close()

    reloaded = SQLiteRuntimeCache("test", config)
    assert reloaded.get_many(["a"]) == [{"value": 1}]
    reloaded.clear()
    assert reloaded.get_many(["a"]) == [None]
    reloaded.close()


def test_null_runtime_cache():
    cache = NullRuntimeCache("test", RuntimeCacheConfig(mode="off"))
    cache.set_many({"a": {"value": 1}})
    assert cache.get_many(["a"]) == [None]
    assert list(cache.items()) == []
