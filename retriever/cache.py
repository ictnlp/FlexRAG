import os
import shelve
from typing import Any

from cachetools import LRUCache


class PersistentLRUCache(LRUCache):
    def __init__(self, persistant_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistant_path = persistant_path
        self._load_cache()

    def _load_cache(self) -> None:
        if os.path.exists(self.persistant_path):
            with shelve.open(self.persistant_path) as db:
                for k, v in db.items():
                    self[k] = v
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
