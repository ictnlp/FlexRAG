from abc import abstractmethod
from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, MutableMapping, Optional

from flexrag.utils import LOGGER_MANAGER

from .backends import STORAGEBACKENDS, StorageBackendConfig
from .serializer import SERIALIZERS, SerializerConfig

logger = LOGGER_MANAGER.get_logger("flexrag.cache")


@dataclass
class PersistentCacheConfig(StorageBackendConfig, SerializerConfig):
    maxsize: Optional[int] = None


class PersistentCacheBase(MutableMapping):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        self.backend = STORAGEBACKENDS.load(cfg)
        self.serializer = SERIALIZERS.load(cfg)
        self._maxsize = cfg.maxsize
        return

    def __getitem__(self, key: Any) -> Any:
        hashed_key = self.hash_key(key)
        if hashed_key in self.backend:
            return self.serializer.deserialize(self.backend[hashed_key])[1]
        raise KeyError(key)

    def __setitem__(self, key: Any, value: Any) -> None:
        hashed_key = self.hash_key(key)
        self.backend[hashed_key] = self.serializer.serialize((key, value))
        self.reduce_size()
        return

    def __delitem__(self, key: Any) -> None:
        hashed_key = self.hash_key(key)
        del self.backend[hashed_key]
        return

    def __len__(self) -> int:
        return len(self.backend)

    def __iter__(self):
        for hashed_key in self.backend:
            key, _ = self.serializer.deserialize(self.backend[hashed_key])
            yield key

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(maxsize={self.maxsize}, currsize={len(self)}) "
            f"{repr(self.backend)}"
        )

    def cache(self, func: callable) -> callable:
        def tupled_args(*args, **kwargs):
            """Return a cache key for the specified hashable arguments."""
            return tuple(args), tuple(sorted(kwargs.items()))

        def wrapper(*args, **kwargs):
            key = tupled_args(*args, **kwargs)
            if key in self:
                return self[key]
            value = func(*args, **kwargs)
            self[key] = value
            return value

        return wrapper

    def __call__(self, func: callable) -> callable:
        return self.cache(func)

    @abstractmethod
    def popitem(self) -> tuple:
        return

    def reduce_size(self, size: int = None) -> None:
        if size is None:
            size = self.maxsize
        while len(self) > size:
            self.popitem()
        return

    def hash_key(self, key: Any) -> bytes:
        """Hash the key."""
        return blake2b(self.serializer.serialize(key)).digest()

    @property
    def maxsize(self) -> int:
        if self._maxsize is None:
            return float("inf")
        return self._maxsize


class RandomPersistentCache(PersistentCacheBase):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        super().__init__(cfg)
        if len(self.backend) > self.maxsize:
            logger.warning(
                "The current cache size is larger than the maxsize."
                "Some items will be evicted."
            )
            self.reduce_size()
        return

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        hashed_key = next(iter(self.backend))
        key, value = self.serializer.deserialize(self.backend.pop(hashed_key))
        return key, value


class LRUPersistentCache(PersistentCacheBase):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        super().__init__(cfg)
        self.order = OrderedDict()
        if len(self.backend) > 0:
            logger.warning(
                "LRUPersistentCache currently does not support loading order from disk."
                "The order will be reset."
            )
            for key in self.backend:
                self.order[key] = None
        if len(self.backend) > self.maxsize:
            logger.warning(
                "The current cache size is larger than the maxsize."
                "Some items will be evicted."
            )
            self.reduce_size()
        return

    def __getitem__(self, key: Any) -> Any:
        self.order.move_to_end(self.hash_key(key))
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.order[self.hash_key(key)] = None
        return super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        del self.order[self.hash_key(key)]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        hashed_key = next(iter(self.order))
        key, value = self.serializer.deserialize(self.backend.pop(hashed_key))
        del self.order[hashed_key]
        return key, value


class LFUPersistentCache(PersistentCacheBase):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        super().__init__(cfg)
        self.counter = Counter()
        if len(self.backend) > 0:
            logger.warning(
                "LFUPersistentCache currently does not support loading counter from disk."
                "The counter will be reset."
            )
            for key in self.backend:
                self.counter[key] = -1
        if len(self.backend) > self.maxsize:
            logger.warning(
                "The current cache size is larger than the maxsize."
                "Some items will be evicted."
            )
            self.reduce_size()
        return

    def __getitem__(self, key: Any) -> Any:
        if self.hash_key(key) in self.backend:
            self.counter[self.hash_key(key)] -= 1
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        hashed_key = self.hash_key(key)
        if hashed_key not in self.backend:
            self.reduce_size(self.maxsize - 1)
        self.counter[hashed_key] = -1
        self.backend[hashed_key] = self.serializer.serialize((key, value))
        return

    def __delitem__(self, key) -> None:
        del self.counter[self.hash_key(key)]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        hashed_key, _ = self.counter.most_common(1)[0]
        key, value = self.serializer.deserialize(self.backend.pop(hashed_key))
        del self.counter[hashed_key]
        return key, value


class FIFOPersistentCache(PersistentCacheBase):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        super().__init__(cfg)
        self.order = OrderedDict()
        if len(self.backend) > 0:
            logger.warning(
                "FIFOPersistentCache currently does not support loading order from disk."
                "The order will be reset."
            )
            for key in self.backend:
                self.order[key] = None
        if len(self.backend) > self.maxsize:
            logger.warning(
                "The current cache size is larger than the maxsize."
                "Some items will be evicted."
            )
            self.reduce_size()
        return

    def __setitem__(self, key, value) -> None:
        self.order[self.hash_key(key)] = None
        return super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        del self.order[self.hash_key(key)]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        hashed_key = next(iter(self.order))
        key, value = self.serializer.deserialize(self.backend.pop(hashed_key))
        del self.order[hashed_key]
        return key, value
