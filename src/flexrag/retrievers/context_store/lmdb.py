from __future__ import annotations

import atexit
import os
from collections.abc import Iterable
from pathlib import Path

import lmdb

from flexrag.common import configure
from flexrag.common.dataclasses import Context
from flexrag.common.serialization import SERIALIZERS, SerializerConfig

from .base import SyncContextStoreBase
from .payload import context_to_payload, payload_to_context


@configure
class LMDBContextStoreConfig:
    """Configuration for ``LMDBContextStore``.

    :param path: Directory path used by the LMDB environment.
    :param serializer: Serializer name registered in FlexRAG serialization.
    :param map_size: Initial LMDB map size in bytes.
    """

    path: str | Path
    serializer: str = "msgpack"
    map_size: int = 1 << 32


class LMDBContextStore(SyncContextStoreBase):
    """LMDB-backed sync-native context store.

    The store persists complete ``Context`` payloads keyed by ``context_id``.
    ``clear`` removes records from the LMDB database but keeps the environment
    directory.
    """

    def __init__(self, config: LMDBContextStoreConfig) -> None:
        """Open or create an LMDB context store.

        :param config: LMDB store configuration.
        """
        self.config = config
        self.path = Path(config.path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.env = lmdb.open(os.fspath(self.path), map_size=config.map_size)
        atexit.register(self.env.close)
        self.serializer = SERIALIZERS.load(SerializerConfig(config.serializer))
        return

    def set_many(self, contexts: Iterable[Context]) -> None:
        items = list(contexts)
        if len({context.context_id for context in items}) != len(items):
            raise ValueError("Context IDs in one batch must be unique.")
        payloads = [
            (
                context.context_id.encode("utf-8"),
                self.serializer.serialize(context_to_payload(context)),
            )
            for context in items
        ]
        try:
            with self.env.begin(write=True) as txn:
                txn.cursor().putmulti(payloads)
        except lmdb.MapFullError:
            self._grow_map_size()
            with self.env.begin(write=True) as txn:
                txn.cursor().putmulti(payloads)
        return

    def get(self, context_id: str) -> Context:
        return self.get_many([context_id])[0]

    def get_many(self, context_ids: Iterable[str]) -> list[Context]:
        ids = [context_id.encode("utf-8") for context_id in context_ids]
        with self.env.begin() as txn:
            values = txn.cursor().getmulti(ids)
        if len(values) != len(ids) or any(value is None for _, value in values):
            raise KeyError("Some context IDs are missing from the context store.")
        return [
            payload_to_context(self.serializer.deserialize(value))
            for _, value in values
        ]

    def iter_contexts(self) -> Iterable[Context]:
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield payload_to_context(self.serializer.deserialize(value))
        return

    @property
    def ids(self) -> list[str]:
        with self.env.begin() as txn:
            return [key.decode("utf-8") for key, _ in txn.cursor()]

    def count(self) -> int:
        return self.env.stat()["entries"]

    def clear(self) -> None:
        db = self.env.open_db()
        with self.env.begin(write=True) as txn:
            txn.drop(db, delete=False)
        return

    def close(self) -> None:
        try:
            atexit.unregister(self.env.close)
        except ValueError:
            pass
        self.env.close()
        return

    def _grow_map_size(self, increment: int = 1 << 30) -> None:
        current_size = self.env.info()["map_size"]
        self.env.set_mapsize(current_size + increment)
        return
