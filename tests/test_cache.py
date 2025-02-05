import os
import tempfile
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from flexrag.cache.backends import (
    DictBackend,
    LMDBBackend,
    LMDBBackendConfig,
    StorageBackendBase,
)
from flexrag.cache.persistent_cache import (
    FIFOPersistentCache,
    LFUPersistentCache,
    LRUPersistentCache,
    PersistentCacheBase,
    PersistentCacheConfig,
    RandomPersistentCache,
)
from flexrag.cache.serializer import (
    CloudPickleSerializer,
    JsonSerializer,
    MsgpackSerializer,
    PickleSerializer,
    SerializerBase,
)


@dataclass
class BackendTestConfig:
    lmdb_config: LMDBBackendConfig = field(default_factory=LMDBBackendConfig)


class TestBackend:
    cfg: BackendTestConfig = OmegaConf.merge(
        OmegaConf.structured(BackendTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "backends.yaml")
        ),
    )

    def backend_test(self, backend: StorageBackendBase):
        # test basic operations
        backend[b"key1"] = b"value1"
        backend[b"key2"] = b"value2"
        backend[b"key3"] = b"value3"
        assert backend.get(b"key1") == b"value1"
        assert backend.get(b"key2") == b"value2"
        assert backend.get(b"key3") == b"value3"
        assert backend[b"key1"] == b"value1"
        assert backend[b"key2"] == b"value2"
        assert backend[b"key3"] == b"value3"
        assert len(backend) == 3
        assert b"key1" in backend
        assert b"key2" in backend
        assert b"key3" in backend
        assert b"key4" not in backend
        assert b"key5" not in backend
        assert b"key6" not in backend

        # test popitem & pop
        key, value = backend.popitem()
        assert len(backend) == 2
        assert isinstance(key, bytes)
        assert isinstance(value, bytes)
        if b"key2" in backend:
            value = backend.pop(b"key2")
            assert len(backend) == 1
            assert value == b"value2"
        else:
            value = backend.pop(b"key1")
            assert len(backend) == 1
            assert value == b"value1"

        # test clear & del
        backend.clear()
        backend[b"key3"] = b"value3"
        backend[b"key4"] = b"value4"
        del backend[b"key3"]
        assert len(backend) == 1
        backend.clear()
        assert len(backend) == 0

        # test update & setdefault
        backend.update({b"key1": b"value1", b"key2": b"value2"})
        assert len(backend) == 2
        assert backend.setdefault(b"key1") == b"value1"
        assert backend.setdefault(b"key3", b"value3") == b"value3"
        assert len(backend) == 3

        # test __iter__ & keys & values & items
        assert set(backend) == {b"key1", b"key2", b"key3"}
        assert set(backend.keys()) == {b"key1", b"key2", b"key3"}
        assert set(backend.values()) == {b"value1", b"value2", b"value3"}
        assert set(backend.items()) == {
            (b"key1", b"value1"),
            (b"key2", b"value2"),
            (b"key3", b"value3"),
        }

        # test __eq__ & __ne__
        dict1 = {b"key1": b"value1", b"key2": b"value2", b"key4": b"value4"}
        dict2 = {b"key1": b"value1", b"key2": b"value2", b"key3": b"value3"}
        assert backend != dict1
        assert backend == dict2
        return

    def test_lmdb_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.cfg.lmdb_config.db_path = os.path.join(tmpdir, "test.lmdb")
            backend = LMDBBackend(self.cfg.lmdb_config)
            self.backend_test(backend)
        return

    def test_dict_backend(self):
        backend = DictBackend()
        self.backend_test(backend)
        return


class TestSerializer:
    def serializer_test(self, serializer: SerializerBase, simple: bool = False):
        # serialize int object
        for i in range(10, 30, 2):
            x = serializer.serialize(i)
            assert serializer.deserialize(x) == i

        # serialize float object
        for i in range(10, 30, 2):
            x = serializer.serialize(float(i) + 0.5)
            assert serializer.deserialize(x) - float(i) - 0.5 < 1e-6

        # serialize str object
        for i in range(10):
            x = serializer.serialize(f"test_{i}")
            assert serializer.deserialize(x) == f"test_{i}"

        # serialize list object
        test_list = [i for i in range(10)]
        x = serializer.serialize(test_list)
        if not simple:
            assert serializer.deserialize(x) == test_list
        else:
            assert list(serializer.deserialize(x)) == test_list

        # serialize dict object
        test_dict = {f"key_{i}": i for i in range(10)}
        x = serializer.serialize(test_dict)
        if not simple:
            assert serializer.deserialize(x) == test_dict
        else:
            assert dict(serializer.deserialize(x)) == test_dict

        # serialize tuple object
        if not simple:
            test_tuple = tuple(i for i in range(10))
            x = serializer.serialize(test_tuple)
            assert serializer.deserialize(x) == test_tuple

        # serialize set object
        if not simple:
            test_set = {i for i in range(10)}
            x = serializer.serialize(test_set)
            assert serializer.deserialize(x) == test_set

        # serialize bytes object
        if not simple:
            test_bytes = b"test_bytes"
            x = serializer.serialize(test_bytes)
            assert serializer.deserialize(x) == test_bytes
        return

    def test_json_serializer(self):
        serializer = JsonSerializer()
        self.serializer_test(serializer, simple=True)
        return

    def test_pickle_serializer(self):
        serializer = PickleSerializer()
        self.serializer_test(serializer)
        return

    def test_cloudpickle_serializer(self):
        serializer = CloudPickleSerializer()
        self.serializer_test(serializer)
        return

    def test_msgpack_serializer(self):
        serializer = MsgpackSerializer()
        self.serializer_test(serializer, simple=True)
        return


class TestPersistentCache:
    cfg: PersistentCacheConfig = PersistentCacheConfig(
        maxsize=3, storage_backend_type="dict", serializer_type="pickle"
    )

    def cache_test(self, cache: PersistentCacheBase):
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

    def test_random_cache(self):
        cache = RandomPersistentCache(self.cfg)
        self.cache_test(cache)
        return

    def test_fifo_cache(self):
        cache = FIFOPersistentCache(self.cfg)
        self.cache_test(cache)

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
        cache = LRUPersistentCache(self.cfg)
        self.cache_test(cache)

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
        cache = LFUPersistentCache(self.cfg)
        self.cache_test(cache)

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
