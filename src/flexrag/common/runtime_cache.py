import json
import os
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Optional

from .default_vars import FLEXRAG_CACHE_DIR
from .logging import LOGGER_MANAGER, warning_once

logger = LOGGER_MANAGER.get_logger("flexrag.runtime_cache")


@dataclass
class RuntimeCacheConfig:
    """Configuration for FlexRAG runtime result caches.

    :param mode: Cache backend mode. ``"off"`` disables caching, ``"memory"``
        keeps results in process memory, and ``"disk"`` stores results in SQLite.
    :type mode: str
    :param cache_dir: Directory used by disk-backed cache implementations.
        Defaults to ``FLEXRAG_CACHE_DIR / "runtime_cache"``. Ignored by in-memory caches.
    :type cache_dir: Path
    :param max_entries: Maximum number of entries kept per namespace. ``None``
        or a negative value means unlimited.
    :type max_entries: Optional[int]
    :param ttl_seconds: Entry time-to-live in seconds, measured from creation
        time. ``None`` disables TTL expiration.
    :type ttl_seconds: Optional[float]
    """

    mode: str = "memory"
    cache_dir: Path = FLEXRAG_CACHE_DIR / "runtime_cache"
    max_entries: Optional[int] = None
    ttl_seconds: Optional[float] = None

    @classmethod
    def from_env(cls) -> "RuntimeCacheConfig":
        """Build runtime cache configuration from environment variables."""
        mode = os.getenv("FLEXRAG_RUNTIME_CACHE_MODE", "memory").strip().lower()
        if mode not in {"off", "memory", "disk"}:
            logger.warning(
                f"Invalid FLEXRAG_RUNTIME_CACHE_MODE={mode!r}; using memory."
            )
            mode = "memory"
        cache_dir = Path(
            os.getenv(
                "FLEXRAG_RUNTIME_CACHE_DIR",
                str(FLEXRAG_CACHE_DIR / "runtime_cache"),
            )
        )
        max_entries = _optional_int_env("FLEXRAG_RUNTIME_CACHE_MAX_ENTRIES")
        ttl_seconds = _optional_float_env("FLEXRAG_RUNTIME_CACHE_TTL_SECONDS")
        return cls(
            mode=mode,
            cache_dir=cache_dir,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )


def make_runtime_cache_key(payload: Any) -> str:
    """Create a stable SHA-256 cache key from a structured payload.

    The payload is normalized into JSON-compatible data before hashing. The
    normalization handles dataclasses, mappings, sets, paths, and array-like
    values so semantically equivalent payloads produce the same key.

    :param payload: Structured data that identifies a runtime cache entry.
    :type payload: Any
    :return: Hex-encoded SHA-256 digest.
    :rtype: str
    """
    data = json.dumps(
        _make_json_serializable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(data.encode("utf-8")).hexdigest()


def _make_json_serializable(data: Any) -> Any:
    if is_dataclass(data) and not isinstance(data, type):
        return _make_json_serializable(asdict(data))
    if isinstance(data, dict):
        return {
            str(k): _make_json_serializable(v)
            for k, v in sorted(data.items(), key=lambda item: str(item[0]))
        }
    if isinstance(data, (list, tuple)):
        return [_make_json_serializable(item) for item in data]
    if isinstance(data, set):
        items = [_make_json_serializable(item) for item in data]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, (str, int, float, bool)) or data is None:
        return data
    if hasattr(data, "tolist"):
        try:
            return _make_json_serializable(data.tolist())
        except Exception:
            pass
    if hasattr(data, "item"):
        try:
            return _make_json_serializable(data.item())
        except Exception:
            pass
    return repr(data)


class RuntimeCache(ABC):
    """Abstract interface for namespaced runtime result caches.

    Runtime caches store ephemeral computation results, such as retrieval
    outputs, behind stable string keys. Implementations are responsible for
    applying TTL and size policies according to ``RuntimeCacheConfig``.

    :param namespace: Logical cache namespace.
    :type namespace: str
    :param config: Runtime cache configuration.
    :type config: RuntimeCacheConfig
    """

    def __init__(self, namespace: str, config: RuntimeCacheConfig) -> None:
        self.namespace = namespace
        self.config = config
        return

    @abstractmethod
    def get_many(self, keys: Iterable[str]) -> list[Any | None]:
        """Return cached values for keys, preserving input order.

        Missing or expired entries are returned as ``None``.
        """
        return

    @abstractmethod
    def set_many(self, items: dict[str, Any], metadata: dict[str, Any] | None = None):
        """Store multiple cache values with optional shared metadata."""
        return

    @abstractmethod
    def clear(self):
        """Remove all entries from this cache namespace."""
        return

    @abstractmethod
    def items(self) -> Iterable[dict[str, Any]]:
        """Iterate over entries in this cache namespace."""
        return

    def close(self):
        """Release resources held by the cache implementation."""
        return


class NullRuntimeCache(RuntimeCache):
    """No-op runtime cache implementation that doesn't store anything."""

    def get_many(self, keys: Iterable[str]) -> list[Any | None]:
        return [None for _ in keys]

    def set_many(self, items: dict[str, Any], metadata: dict[str, Any] | None = None):
        return

    def clear(self):
        return

    def items(self) -> Iterable[dict[str, Any]]:
        return []


class MemoryRuntimeCache(RuntimeCache):
    """In-memory runtime cache implementation that keeps entries in process memory."""

    def __init__(self, namespace: str, config: RuntimeCacheConfig) -> None:
        super().__init__(namespace, config)
        self._items: dict[str, dict[str, Any]] = {}
        return

    def get_many(self, keys: Iterable[str]) -> list[Any | None]:
        now = time.time()
        results = []
        for key in keys:
            item = self._items.get(key)
            if item is None:
                results.append(None)
                continue
            if self._is_expired(item, now):
                self._items.pop(key, None)
                results.append(None)
                continue
            item["accessed_at"] = now
            results.append(item["value"])
        return results

    def set_many(self, items: dict[str, Any], metadata: dict[str, Any] | None = None):
        now = time.time()
        for key, value in items.items():
            self._items[key] = {
                "key": key,
                "value": value,
                "metadata": metadata or {},
                "created_at": now,
                "accessed_at": now,
                "size_bytes": len(pickle.dumps(value)),
            }
        self._prune()
        return

    def clear(self):
        self._items.clear()
        return

    def items(self) -> Iterable[dict[str, Any]]:
        now = time.time()
        expired = []
        for key, item in self._items.items():
            if self._is_expired(item, now):
                expired.append(key)
                continue
            yield dict(item)
        for key in expired:
            self._items.pop(key, None)
        return

    def _is_expired(self, item: dict[str, Any], now: float) -> bool:
        ttl = self.config.ttl_seconds
        return ttl is not None and now - item["created_at"] > ttl

    def _prune(self):
        max_entries = self.config.max_entries
        if max_entries is None or max_entries < 0:
            return
        while len(self._items) > max_entries:
            key = min(
                self._items,
                key=lambda k: (
                    self._items[k]["accessed_at"],
                    self._items[k]["created_at"],
                ),
            )
            self._items.pop(key, None)
        return


class SQLiteRuntimeCache(RuntimeCache):
    """Disk-backed runtime cache implementation that stores entries in a SQLite database."""

    def __init__(self, namespace: str, config: RuntimeCacheConfig) -> None:
        super().__init__(namespace, config)
        self._conn: sqlite3.Connection | None = None
        return

    def get_many(self, keys: Iterable[str]) -> list[Any | None]:
        keys = list(keys)
        if not keys:
            return []
        conn = self._connect()
        now = time.time()
        self._delete_expired(now)
        placeholders = ",".join("?" for _ in keys)
        rows = conn.execute(
            f"""
            SELECT key, value FROM cache_entries
            WHERE namespace = ? AND key IN ({placeholders})
            """,
            [self.namespace, *keys],
        ).fetchall()
        values = {key: pickle.loads(value) for key, value in rows}
        hit_keys = list(values)
        if hit_keys:
            placeholders = ",".join("?" for _ in hit_keys)
            conn.execute(
                f"""
                UPDATE cache_entries SET accessed_at = ?
                WHERE namespace = ? AND key IN ({placeholders})
                """,
                [now, self.namespace, *hit_keys],
            )
            conn.commit()
        return [values.get(key) for key in keys]

    def set_many(self, items: dict[str, Any], metadata: dict[str, Any] | None = None):
        if not items:
            return
        conn = self._connect()
        now = time.time()
        encoded_metadata = json.dumps(metadata or {}, ensure_ascii=False)
        rows = []
        for key, value in items.items():
            payload = pickle.dumps(value)
            rows.append(
                (
                    self.namespace,
                    key,
                    payload,
                    encoded_metadata,
                    now,
                    now,
                    len(payload),
                )
            )
        conn.executemany(
            """
            INSERT INTO cache_entries (
                namespace, key, value, metadata,
                created_at, accessed_at, size_bytes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(namespace, key) DO UPDATE SET
                value = excluded.value,
                metadata = excluded.metadata,
                created_at = excluded.created_at,
                accessed_at = excluded.accessed_at,
                size_bytes = excluded.size_bytes
            """,
            rows,
        )
        conn.commit()
        self._prune()
        return

    def clear(self):
        conn = self._connect()
        conn.execute(
            "DELETE FROM cache_entries WHERE namespace = ?",
            (self.namespace,),
        )
        conn.commit()
        return

    def items(self) -> Iterable[dict[str, Any]]:
        conn = self._connect()
        self._delete_expired(time.time())
        rows = conn.execute(
            """
            SELECT key, value, metadata, created_at, accessed_at, size_bytes
            FROM cache_entries
            WHERE namespace = ?
            ORDER BY created_at ASC
            """,
            (self.namespace,),
        ).fetchall()
        for key, value, metadata, created_at, accessed_at, size_bytes in rows:
            yield {
                "key": key,
                "value": pickle.loads(value),
                "metadata": json.loads(metadata or "{}"),
                "created_at": created_at,
                "accessed_at": accessed_at,
                "size_bytes": size_bytes,
            }
        return

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        return

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = self.config.cache_dir / "runtime_cache.sqlite3"
            self._conn = sqlite3.connect(db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value BLOB NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(namespace, key)
                )
                """
            )
            self._conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cache_entries_accessed
                ON cache_entries(namespace, accessed_at, created_at)
                """
            )
            self._conn.commit()
        return self._conn

    def _delete_expired(self, now: float):
        ttl = self.config.ttl_seconds
        if ttl is None:
            return
        conn = self._connect()
        conn.execute(
            """
            DELETE FROM cache_entries
            WHERE namespace = ? AND created_at < ?
            """,
            (self.namespace, now - ttl),
        )
        conn.commit()
        return

    def _prune(self):
        max_entries = self.config.max_entries
        if max_entries is None or max_entries < 0:
            return
        conn = self._connect()
        total = conn.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE namespace = ?",
            (self.namespace,),
        ).fetchone()[0]
        extra = total - max_entries
        if extra <= 0:
            return
        conn.execute(
            """
            DELETE FROM cache_entries
            WHERE namespace = ? AND key IN (
                SELECT key FROM cache_entries
                WHERE namespace = ?
                ORDER BY accessed_at ASC, created_at ASC
                LIMIT ?
            )
            """,
            (self.namespace, self.namespace, extra),
        )
        conn.commit()
        return


_RUNTIME_CACHES: dict[str, RuntimeCache] = {}


def get_runtime_cache(namespace: str) -> RuntimeCache:
    if namespace not in _RUNTIME_CACHES:
        config = RuntimeCacheConfig.from_env()
        _RUNTIME_CACHES[namespace] = _create_cache(namespace, config)
    return _RUNTIME_CACHES[namespace]


def reset_runtime_caches():
    for cache in _RUNTIME_CACHES.values():
        cache.close()
    _RUNTIME_CACHES.clear()
    return


def _create_cache(namespace: str, config: RuntimeCacheConfig) -> RuntimeCache:
    if config.mode == "off":
        return NullRuntimeCache(namespace, config)
    if config.mode == "memory":
        return MemoryRuntimeCache(namespace, config)
    try:
        cache = SQLiteRuntimeCache(namespace, config)
        cache._connect()
        return cache
    except Exception as e:
        warning_once(
            logger,
            "Failed to initialize SQLite runtime cache; falling back to memory: %s",
            e,
        )
        return MemoryRuntimeCache(namespace, config)


def _optional_int_env(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid {name}={value!r}; ignoring it.")
        return None


def _optional_float_env(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid {name}={value!r}; ignoring it.")
        return None
