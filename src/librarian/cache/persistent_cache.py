from collections import Counter, OrderedDict
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any, MutableMapping, Optional, Literal

from librarian.utils import Choices, LOGGER_MANAGER

from .backends import PersistentBackendConfig, load_backend

logger = LOGGER_MANAGER.get_logger("librarian.cache")


def tupled_hashkey(*args, **kwargs):
    """Return a cache key for the specified hashable arguments."""
    return tuple(args), tuple(sorted(kwargs.items()))


@dataclass
class PersistentCacheConfig(PersistentBackendConfig):
    maxsize: Optional[int] = None
    evict_order: Choices(["LRU", "LFU", "FIFO"]) = "LRU"  # type: ignore
    reset_arguments: bool = False


class PersistentCache(MutableMapping):
    def __init__(self, cfg: PersistentCacheConfig) -> None:
        self.__backend = load_backend(cfg)

        # set basic arguments
        if cfg.maxsize is None:
            logger.warning("maxsize is not set. Set to infinity.")
            maxsize = float("inf")
        else:
            maxsize = cfg.maxsize
        self.__meta_key = blake2b("meta".encode()).hexdigest()
        meta_data = self.__backend.get(
            self.__meta_key,
            {
                "maxsize": maxsize,
                "evict_order": cfg.evict_order,
                "order": OrderedDict(),
                "counter": Counter(),
            },
        )
        self.__backend[self.__meta_key] = meta_data
        if cfg.reset_arguments:
            self.reset_arguments(maxsize, cfg.evict_order)

        # check consistency
        self.__check(fully_check=True)
        return

    def __getitem__(self, key) -> Any:
        if key in self.__backend:
            self.__update(key, action="get")
            return self.__backend[key]
        raise KeyError(key)

    def __setitem__(self, key, value: Any) -> None:
        self.__update(key, action="set")
        self.__backend[key] = value
        while len(self) > self.maxsize:
            self.popitem()
        return

    def __delitem__(self, key) -> None:
        del self.__backend[key]
        self.__update(key, action="delete")
        return

    def __len__(self) -> int:
        return len(self.__backend) - 1

    def __iter__(self):
        meta_key = blake2b("meta".encode()).hexdigest()
        for key in self.__backend:
            if key == meta_key:
                continue
            yield key

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(evict_order={self.evict_order}, maxsize={self.maxsize}, currsize={self.currsize}) {repr(self.__backend)}"

    def __call__(self, func: callable) -> callable:
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

    def popitem(self) -> tuple:
        if not self:
            raise KeyError("popitem(): cache is empty")
        meta = self.__backend[self.__meta_key]
        meta["counter"] = Counter(meta["counter"])
        meta["order"] = OrderedDict(meta["order"])
        match meta["evict_order"]:
            case "LRU":
                key, _ = meta["order"].popitem(last=False)
            case "LFU":
                key, _ = meta["counter"].most_common(1)[0]
            case "FIFO":
                key, _ = meta["order"].popitem(last=False)
        value = self.__backend.pop(key)
        del meta["counter"][key]
        self.__backend[self.__meta_key] = meta
        return key, value

    def reset_arguments(self, maxsize: int, evict_order: str = None) -> None:
        meta = self.__backend[self.__meta_key]
        meta["maxsize"] = maxsize
        meta["evict_order"] = evict_order or meta["evict_order"]
        while self.currsize > self.maxsize:
            self.popitem()
        return

    def __update(self, key, action: Literal["delete", "get", "set"]) -> None:
        meta = self.__backend[self.__meta_key]
        meta["counter"] = Counter(meta["counter"])
        meta["order"] = OrderedDict(meta["order"])
        match action:
            case "delete":
                if key in meta["order"]:
                    del meta["order"][key]
                if key in meta["counter"]:
                    del meta["counter"][key]
            case "get":
                if (meta["evict_order"] == "LRU") and (key in meta["order"]):
                    meta["order"].move_to_end(key)
                meta["counter"][key] -= 1
            case "set":
                if key in meta["order"]:
                    meta["order"].move_to_end(key)
                else:
                    meta["order"][key] = None
                meta["counter"][key] -= 1
        self.__backend[self.__meta_key] = meta  # write back
        return

    def __check(self, fully_check: bool = False) -> None:
        """Check the consistency of the cache."""
        meta = self.__backend[self.__meta_key]
        assert len(meta["order"]) == len(self)
        assert len(meta["counter"]) == len(self)
        if fully_check:
            keys = set(self.__backend.keys())
            meta_key = blake2b("meta".encode()).hexdigest()
            assert set(meta["order"].keys()) == keys - {meta_key}
            assert set(meta["counter"].keys()) == keys - {meta_key}
        return

    @property
    def maxsize(self) -> int:
        """The maximum size of the cache."""
        return self.__backend[self.__meta_key]["maxsize"]

    @property
    def currsize(self) -> int:
        """The current size of the cache."""
        return len(self)

    @property
    def evict_order(self) -> str:
        """The eviction order of the cache."""
        return self.__backend[self.__meta_key]["evict_order"]
