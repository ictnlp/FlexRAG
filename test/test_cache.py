import os
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kylin.cache import (
    PersistentCacheConfig,
    PersistentCache,
    LMDBBackendConfig,
    ShelveBackendConfig,
)


class TestCache:
    def test_lmdb_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lmdb_cfg = PersistentCacheConfig(
                backend="lmdb",
                lmdb_config=LMDBBackendConfig(
                    db_path=os.path.join(tmpdir, "test.lmdb"),
                    serializer="pickle",
                ),
                maxsize=3,
                evict_order="LRU",
                reset_arguments=False,
            )
            cache = PersistentCache(lmdb_cfg)

            # test __setitem__ & __len__ & keys & values & __getitem__
            cache["a"] = "a"
            cache["b"] = "b"
            cache["c"] = "c"
            assert cache["a"] == "a"
            cache["d"] = "d"
            assert len(cache) == 3
            assert set(cache.keys()) == {"d", "a", "c"}
            assert set(cache.values()) == {"d", "a", "c"}

            # test __delitem__
            del cache["c"]
            assert len(cache) == 2

            # test __contains__
            assert "a" in cache
            assert "b" not in cache

            # test __iter__
            assert set(cache) == {"d", "a"}

            # test loading
            del cache
            cache = PersistentCache(lmdb_cfg)
            assert len(cache) == 2

            # test clear
            cache.clear()
            assert len(cache) == 0
        return

    def test_shelve_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PersistentCacheConfig(
                backend="shelve",
                shelve_config=ShelveBackendConfig(
                    db_path=os.path.join(tmpdir, "test.shelve"),
                ),
                maxsize=3,
                evict_order="LRU",
                reset_arguments=False,
            )
            cache = PersistentCache(cfg)

            # test __setitem__ & __len__ & keys & values & __getitem__
            cache["a"] = "a"
            cache["b"] = "b"
            cache["c"] = "c"
            assert cache["a"] == "a"
            cache["d"] = "d"
            assert len(cache) == 3
            assert set(cache.keys()) == {"d", "a", "c"}
            assert set(cache.values()) == {"d", "a", "c"}

            # test __delitem__
            del cache["c"]
            assert len(cache) == 2

            # test __contains__
            assert "a" in cache
            assert "b" not in cache

            # test __iter__
            assert set(cache) == {"d", "a"}

            # test loading
            del cache
            cache = PersistentCache(cfg)
            assert len(cache) == 2

            # test clear
            cache.clear()
            assert len(cache) == 0
        return
