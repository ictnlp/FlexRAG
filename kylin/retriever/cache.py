import json
import os
import shelve
import sqlite3
from collections import OrderedDict
from hashlib import md5
from typing import Any, MutableMapping

from cachetools import LRUCache


class PersistentLRUCache(LRUCache):
    def __init__(self, persistant_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.persistant_path = persistant_path
        self._load_cache()
        return

    def _load_cache(self) -> None:
        if not os.path.exists(os.path.dirname(self.persistant_path)):
            os.makedirs(os.path.dirname(self.persistant_path))
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

    def clean(self):
        return super().clear()


class SQLLRUCache(MutableMapping):
    """A simple LRU cache using SQLite3 as a persistent backend."""

    def __init__(self, persistant_path: str, maxsize: int):
        self._maxsize = maxsize
        self._persistant_path = os.path.abspath(persistant_path)
        self._init_cache()
        return

    def _init_cache(self) -> None:
        if not os.path.exists(os.path.dirname(self._persistant_path)):
            os.makedirs(os.path.dirname(self._persistant_path))
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    cache_order INTEGER,
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL
                )
            """
            )
            conn.commit()
        return

    def __getitem__(self, key) -> None:
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM cache WHERE key = ?",
                (key,),
            )
            value = cursor.fetchone()
            if value is not None:
                value = value[0]
            else:
                raise KeyError(key)
        self._move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        while len(self) >= self._maxsize:
            self.popitem()
        order = len(self)
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                (order, key, value),
            )
            conn.commit()
        return

    def __delitem__(self, key):
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cache_order FROM cache WHERE key = ?", (key,))
            order = cursor.fetchone()[0]
            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
            cursor.execute(
                "UPDATE cache SET cache_order = cache_order - 1 WHERE cache_order > ?",
                (order,),
            )
            conn.commit()
        return

    def __len__(self) -> int:
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cache")
            size = cursor.fetchone()[0]
        return size

    def __contains__(self, key) -> bool:
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM cache WHERE key = ?",
                (key,),
            )
            return cursor.fetchone() is not None

    def __iter__(self):
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key FROM cache ORDER BY cache_order")
            keys = cursor.fetchall()
            keys = [key[0] for key in keys]
        return iter(keys)

    def to_dict(self) -> OrderedDict:
        """Return a new view of the cache as an ordered dictionary."""
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM cache ORDER BY cache_order")
            items = cursor.fetchall()
        return OrderedDict(items)

    @property
    def maxsize(self):
        """The maximum size of the cache."""
        return self._maxsize

    def _move_to_end(self, key):
        end_order = len(self) - 1
        with sqlite3.connect(self._persistant_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cache_order FROM cache WHERE key = ?", (key,))
            order = cursor.fetchone()[0]
            cursor.execute(
                "UPDATE cache SET cache_order = cache_order - 1 WHERE cache_order > ?",
                (order,),
            )
            cursor.execute(
                "UPDATE cache SET cache_order = ? WHERE key = ?",
                (end_order, key),
            )
            conn.commit()
        return

    def clean(self):
        return super().clear()


def hashkey(*args, **kwargs) -> str:
    """Return a cache key for the specified hashable arguments."""
    key = json.dumps([args, sorted(kwargs.items())], sort_keys=True)
    key = md5(key.encode("utf-8")).hexdigest()
    return key
