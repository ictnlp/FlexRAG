import json
import os
import shelve
from hashlib import md5
from typing import Any

from cachetools import LRUCache


class PersistentLRUCache(LRUCache):
    def __init__(self, persistant_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistant_path = persistant_path
        self._load_cache()
        return

    def _load_cache(self) -> None:
        if os.path.exists(self.persistant_path + ".dat"):
            with shelve.open(self.persistant_path) as db:
                for k, v in db.items():
                    super().__setitem__(k, v)
        return

    def __setitem__(self, key: Any, value: Any) -> None:
        with shelve.open(self.persistant_path) as db:
            db[key] = value
        return super().__setitem__(key, value)

    def __delitem__(self, key: Any) -> None:
        with shelve.open(self.persistant_path) as db:
            if key in db:
                del db[key]
        return super().__delitem__(key)


def hashkey(*args, **kwargs) -> str:
    """Return a cache key for the specified hashable arguments."""
    key = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
    key = md5(key.encode("utf-8")).hexdigest()
    return key
