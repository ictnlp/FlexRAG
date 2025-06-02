from flexrag.utils.persistent_cache import (
    FIFOPersistentCache,
    LFUPersistentCache,
    LRUPersistentCache,
    PersistentCacheBase,
    RandomPersistentCache,
)


class TestPersistentCache:

    def run_cache(self, cache: PersistentCacheBase):
        # prepare cached function
        cache.clear()
        called_args = set()

        @cache.cache
        def test_func(a, b):
            if (a, b) in called_args:
                return "missed"
            called_args.add((a, b))
            return a + b

        # fill the cache
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5

        # test cache hit
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5

        # test cache miss
        cache.clear()
        assert test_func(1, 2) == "missed"
        assert test_func(1, 3) == "missed"
        assert test_func(1, 4) == "missed"
        return

    def run_mapping(self, cache: PersistentCacheBase):
        cache.clear()
        # test `__set__` operation
        cache["key1"] = "value1"
        cache["key2"] = ["value", 2]
        cache["key3"] = {"data": "value3"}

        # test `get` operation
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == ["value", 2]
        assert cache.get("key3") == {"data": "value3"}

        # test `__get__` operation
        assert cache["key1"] == "value1"
        assert cache["key2"] == ["value", 2]
        assert cache["key3"] == {"data": "value3"}

        # test `__len__` operation
        assert len(cache) == 3

        # test `__contains__` operation
        assert "key1" in cache
        assert "key2" in cache
        assert "key3" in cache
        assert "key4" not in cache
        assert "key5" not in cache
        assert "key6" not in cache

        # test popitem & pop
        key, value = cache.popitem()
        assert len(cache) == 2
        assert isinstance(key, str)
        if "key2" in cache:
            value = cache.pop("key2")
            assert len(cache) == 1
            assert value == ["value", 2]
        else:
            value = cache.pop("key1")
            assert len(cache) == 1
            assert value == "value1"

        # test clear & del
        cache.clear()
        cache["key3"] = {"data": "value3"}
        cache["key4"] = {"data": "value4"}
        del cache["key3"]
        assert len(cache) == 1
        cache.clear()
        assert len(cache) == 0

        # test update & setdefault
        cache.update({"key1": "value1", "key2": ["value", 2]})
        assert len(cache) == 2
        assert cache.setdefault("key1") == "value1"
        assert cache.setdefault("key3", {"data": "value3"}) == {"data": "value3"}
        assert len(cache) == 3

        # test __iter__ & keys & values & items
        assert set(cache) == {"key1", "key2", "key3"}
        assert set(cache.keys()) == {"key1", "key2", "key3"}
        expected_values = [
            "value1",
            ["value", 2],
            {"data": "value3"},
        ]
        actual_values = cache.values()
        for actual_value in actual_values:
            assert actual_value in expected_values
            expected_values.remove(actual_value)
        assert len(expected_values) == 0
        expected_items = [
            ("key1", "value1"),
            ("key2", ["value", 2]),
            ("key3", {"data": "value3"}),
        ]
        actual_items = cache.items()
        for actual_item in actual_items:
            assert actual_item in expected_items
            expected_items.remove(actual_item)
        assert len(expected_items) == 0

        # test __eq__ & __ne__
        dict1 = {
            "key1": "value1",
            "key2": ["value", 2],
            "key4": {"data": "value4"},
        }
        dict2 = {
            "key1": "value1",
            "key2": ["value", 2],
            "key3": {"data": "value3"},
        }
        assert cache != dict1
        assert cache == dict2
        return

    def test_random_cache(self):
        cache = RandomPersistentCache(maxsize=3)
        self.run_mapping(cache)
        self.run_cache(cache)
        return

    def test_fifo_cache(self):
        cache = FIFOPersistentCache(maxsize=3)
        self.run_mapping(cache)
        self.run_cache(cache)

        # test fifo eviction
        cache.clear()
        called_args = set()

        @cache.cache
        def test_func(a, b):
            if (a, b) in called_args:
                return "missed"
            called_args.add((a, b))
            return a + b

        # fill the cache
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        # test fifo eviction
        assert test_func(1, 5) == 6
        assert test_func(1, 2) == "missed"
        assert test_func(1, 6) == 7
        assert test_func(1, 3) == "missed"
        assert test_func(1, 7) == 8
        assert test_func(1, 4) == "missed"
        return

    def test_lru_cache(self):
        cache = LRUPersistentCache(maxsize=3)
        self.run_mapping(cache)
        self.run_cache(cache)

        # test lru eviction
        called_args = set()

        @cache.cache
        def test_func(a, b):
            if (a, b) in called_args:
                return "missed"
            called_args.add((a, b))
            return a + b

        # case 1
        called_args.clear()
        cache.clear()
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        assert test_func(1, 2) == 3
        assert test_func(1, 5) == 6
        assert test_func(1, 3) == "missed"

        # case 2
        called_args.clear()
        cache.clear()
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        assert test_func(1, 3) == 4
        assert test_func(1, 6) == 7
        assert test_func(1, 2) == "missed"
        assert test_func(1, 4) == "missed"
        assert test_func(1, 6) == 7
        return

    def test_lfu_cache(self):
        cache = LFUPersistentCache(maxsize=3)
        self.run_mapping(cache)
        self.run_cache(cache)

        # test lfu eviction
        called_args = set()

        @cache.cache
        def test_func(a, b):
            if (a, b) in called_args:
                return "missed"
            called_args.add((a, b))
            return a + b

        # case 1
        called_args.clear()
        cache.clear()
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 5) == 6
        assert test_func(1, 4) == "missed"

        # case 2
        called_args.clear()
        cache.clear()
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 5) == 6
        assert test_func(1, 6) == 7
        assert test_func(1, 5) == "missed"

        # case 3
        called_args.clear()
        cache.clear()
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 4) == 5
        assert test_func(1, 2) == 3
        assert test_func(1, 3) == 4
        assert test_func(1, 5) == 6
        assert test_func(1, 6) == 7
        assert test_func(1, 6) == 7
        assert test_func(1, 6) == 7
        assert test_func(1, 2) == 3
        assert test_func(1, 7) == 8
        assert test_func(1, 3) == "missed"
        return
