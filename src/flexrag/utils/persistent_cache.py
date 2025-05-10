from abc import abstractmethod
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, MutableMapping, Optional

from flexrag.database import LMDBRetrieverDatabase, NaiveRetrieverDatabase

from .logging import LOGGER_MANAGER

logger = LOGGER_MANAGER.get_logger("flexrag.cache")


@dataclass
class PersistentCacheConfig:
    """Configuration for the PersistentCacheBase class.

    :param maxsize: The maximum size of the cache. If None, the cache will not be limited.
    :type maxsize: Optional[int]
    :param cache_path: The path to the cache file. If None, the cache will not be persisted.
    :type cache_path: Optional[str]
    """

    maxsize: Optional[int] = None
    cache_path: Optional[str] = None


class PersistentCacheBase(MutableMapping[str, Any]):
    """The base class for PersistentCache.

    The PersistentCache is a cache that can be persisted to disk,
    and provide a simple interface like a dictionary.
    The subclasses should implement the popitem method,
    which decides which item to evict from the cache when the cache is full.
    """

    def __init__(
        self, maxsize: Optional[int] = None, cache_path: Optional[str] = None
    ) -> None:
        if cache_path is None:
            self.backend = NaiveRetrieverDatabase()
        else:
            self.backend = LMDBRetrieverDatabase(cache_path)
        self._maxsize = maxsize
        return

    def __getitem__(self, key: str) -> Any:
        return self.backend[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self.backend[key] = value
        self.reduce_size()
        return

    def __delitem__(self, key: str) -> None:
        del self.backend[key]
        return

    def __len__(self) -> int:
        return len(self.backend)

    def __iter__(self):
        return self.backend.__iter__()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(maxsize={self.maxsize}, currsize={len(self)}) "
            f"{repr(self.backend)}"
        )

    def cache(self, func: callable) -> callable:
        """Decorator to cache the result of a function.
        The arguments of the function should be hashable.

        For example:

        .. code-block:: python

            from flexrag.utils import LRUPersistentCache

            cache = LRUPersistentCache()

            @cache.cache
            def expensive_function(x):
                # Some expensive computation
                return x * 2
        """

        def tupled_args(*args, **kwargs):
            """Return a cache key for the specified hashable arguments."""
            return tuple(args), tuple(sorted(kwargs.items()))

        def wrapper(*args, **kwargs):
            key = str(hash(tupled_args(*args, **kwargs)))
            if key in self:
                return self[key]
            value = func(*args, **kwargs)
            self[key] = value
            return value

        return wrapper

    def __call__(self, func: callable) -> callable:
        """Decorator to cache the result of a function.
        This is a shorthand for ``cache.cache(func)``.

        For example:

        .. code-block:: python

            from flexrag.utils import LRUPersistentCache

            cache = LRUPersistentCache()

            @cache
            def expensive_function(x):
                # Some expensive computation
                return x * 2
        """
        return self.cache(func)

    @abstractmethod
    def popitem(self) -> tuple:
        """This method should be implemented by subclasses."""
        return

    def reduce_size(self, size: int = None) -> None:
        """Reduce the size of the cache to the specified size.

        param size: The size to reduce to. If None, use the self.maxsize.
        type size: int
        return: None
        rtype: None
        """
        if size is None:
            size = self.maxsize
        while len(self) > size:
            self.popitem()
        return

    @property
    def maxsize(self) -> int:
        if self._maxsize is None:
            return 1 << 31
        return self._maxsize


class RandomPersistentCache(PersistentCacheBase):
    """
    The RandomPersistentCache evicts a random item from the cache when the cache is full.

    In this implementation, the evict order is determined by the __iter__ method of the backend.
    """

    def __init__(
        self, maxsize: Optional[int] = None, cache_path: Optional[str] = None
    ) -> None:
        super().__init__(maxsize, cache_path)
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
        evict_key = next(iter(self.backend))
        value = self.backend.pop(evict_key)
        return evict_key, value


class LRUPersistentCache(PersistentCacheBase):
    """The LRUPersistentCache evicts the least recently used item from the cache when the cache is full.

    This implementation employs an OrderedDict to keep track of the order of access.
    However, the order will not be persisted to disk.
    Thus, the order will be reset when the cache is loaded from disk.
    """

    def __init__(
        self, maxsize: Optional[int] = None, cache_path: Optional[str] = None
    ) -> None:
        super().__init__(maxsize, cache_path)
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
        self.order.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        self.order[key] = None
        return super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        del self.order[key]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        evict_key = next(iter(self.order))
        value = self.backend.pop(evict_key)
        del self.order[evict_key]
        return evict_key, value


class LFUPersistentCache(PersistentCacheBase):
    """The LFUPersistentCache evicts the least frequently used item from the cache when the cache is full.

    This implementation employs a Counter to keep track of the frequency of access.
    However, the frequency will not be persisted to disk.
    Thus, the frequency will be reset when the cache is loaded from disk.
    """

    def __init__(
        self, maxsize: Optional[int] = None, cache_path: Optional[str] = None
    ) -> None:
        super().__init__(maxsize, cache_path)
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
        if key in self.backend:
            self.counter[key] -= 1
        return super().__getitem__(key)

    def __setitem__(self, key, value) -> None:
        if key not in self.backend:
            self.reduce_size(self.maxsize - 1)
        self.counter[key] = -1
        self.backend[key] = value
        return

    def __delitem__(self, key) -> None:
        del self.counter[key]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        evict_key, _ = self.counter.most_common(1)[0]
        value = self.backend.pop(evict_key)
        del self.counter[evict_key]
        return evict_key, value


class FIFOPersistentCache(PersistentCacheBase):
    def __init__(
        self, maxsize: Optional[int] = None, cache_path: Optional[str] = None
    ) -> None:
        super().__init__(maxsize, cache_path)
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
        self.order[key] = None
        return super().__setitem__(key, value)

    def __delitem__(self, key) -> None:
        del self.order[key]
        return super().__delitem__(key)

    def popitem(self) -> tuple:
        if len(self) == 0:
            raise KeyError("popitem(): cache is empty")
        evict_key = next(iter(self.order))
        value = self.backend.pop(evict_key)
        del self.order[evict_key]
        return evict_key, value
