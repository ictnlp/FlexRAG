from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from threading import RLock

from flexrag.common import configure
from flexrag.common.dataclasses import Context
from flexrag.common.serialization import SERIALIZERS, SerializerConfig

from .base import SyncContextStoreBase
from .payload import context_to_payload, payload_to_context


@configure
class SQLiteContextStoreConfig:
    """Configuration for ``SQLiteContextStore``.

    :param path: SQLite database file path.
    :param serializer: Serializer name registered in FlexRAG serialization.
    :param table_name: Table name used for context payload rows.
    :param timeout: SQLite connection timeout in seconds.
    """

    path: str | Path
    serializer: str = "msgpack"
    table_name: str = "contexts"
    timeout: float = 30.0


class SQLiteContextStore(SyncContextStoreBase):
    """SQLite-backed sync-native context store.

    The store persists complete ``Context`` payloads in one table keyed by
    ``context_id``. ``clear`` deletes table rows but keeps the database file.
    """

    def __init__(self, config: SQLiteContextStoreConfig) -> None:
        """Open or create a SQLite context store.

        :param config: SQLite store configuration.
        """
        self.config = config
        self.path = Path(config.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.table_name = self._validate_table_name(config.table_name)
        self._lock = RLock()
        self.conn = sqlite3.connect(
            os.fspath(self.path),
            timeout=config.timeout,
            check_same_thread=False,
        )
        self.serializer = SERIALIZERS.load(SerializerConfig(config.serializer))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
            "context_id TEXT PRIMARY KEY, "
            "payload BLOB NOT NULL)"
        )
        self.conn.commit()
        return

    def set_many(self, contexts: Iterable[Context]) -> None:
        items = list(contexts)
        if len({context.context_id for context in items}) != len(items):
            raise ValueError("Context IDs in one batch must be unique.")
        payloads = [
            (
                context.context_id,
                sqlite3.Binary(
                    self.serializer.serialize(context_to_payload(context))
                ),
            )
            for context in items
        ]
        with self._lock, self.conn:
            self.conn.executemany(
                f"INSERT OR REPLACE INTO {self.table_name} "
                "(context_id, payload) VALUES (?, ?)",
                payloads,
            )
        return

    def get(self, context_id: str) -> Context:
        return self.get_many([context_id])[0]

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        ids = [context_id for context_id in context_ids]
        if not ids:
            return []
        placeholders = ", ".join("?" for _ in ids)
        with self._lock:
            rows = self.conn.execute(
                f"SELECT context_id, payload FROM {self.table_name} "
                f"WHERE context_id IN ({placeholders})",
                ids,
            ).fetchall()
        payload_by_id = {context_id: payload for context_id, payload in rows}
        if any(context_id not in payload_by_id for context_id in ids):
            raise KeyError("Some context IDs are missing from the context store.")
        return [
            payload_to_context(self.serializer.deserialize(payload_by_id[context_id]))
            for context_id in ids
        ]

    def iter_contexts(self) -> Iterable[Context]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT payload FROM {self.table_name} ORDER BY context_id"
            ).fetchall()
        for (payload,) in rows:
            yield payload_to_context(self.serializer.deserialize(payload))
        return

    @property
    def ids(self) -> list[str]:
        with self._lock:
            rows = self.conn.execute(
                f"SELECT context_id FROM {self.table_name} ORDER BY context_id"
            ).fetchall()
        return [context_id for (context_id,) in rows]

    def count(self) -> int:
        with self._lock:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self.table_name}"
            ).fetchone()
        return int(row[0])

    def clear(self) -> None:
        with self._lock, self.conn:
            self.conn.execute(f"DELETE FROM {self.table_name}")
        return

    def close(self) -> None:
        with self._lock:
            self.conn.close()
        return

    @staticmethod
    def _validate_table_name(table_name: str) -> str:
        if not table_name:
            raise ValueError("table_name must be non-empty.")
        if not table_name.replace("_", "").isalnum() or table_name[0].isdigit():
            raise ValueError(
                "table_name must contain only letters, numbers, and underscores, "
                "and must not start with a digit."
            )
        return table_name
