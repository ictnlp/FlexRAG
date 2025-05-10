import atexit
import os

import lmdb
import numpy as np

from .databsae_base import RetrieverDatabaseBase
from .serializer import SERIALIZERS, SerializerConfig


class LMDBRetrieverDatabase(RetrieverDatabaseBase):
    """A RetrieverDatabase that uses LMDB as the backend storage format."""

    def __init__(
        self,
        database_path: str,
        map_size: int = 1 << 40,
        serializer="msgpack",
    ) -> None:
        super().__init__()

        # prepare database path
        self.database_path = database_path
        if not os.path.exists(database_path):
            os.makedirs(database_path)

        # open database
        self.database = lmdb.open(database_path, map_size=map_size)
        atexit.register(self.database.close)

        # prepare serializer
        self.serializer = SERIALIZERS.load(SerializerConfig(serializer))
        return

    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        if isinstance(idx, str):
            ids = [idx]
        else:
            ids = idx

        items = []
        with self.database.begin() as txn:
            for id_ in ids:
                item = txn.get(id_.encode("utf-8"))
                if item is None:
                    raise KeyError(id_)
                items.append(self.serializer.deserialize(item))

        if isinstance(idx, str):
            return items[0]
        return items

    def __setitem__(
        self, ids: str | list[str] | np.ndarray, data: dict | list[dict]
    ) -> None:
        if isinstance(ids, str):
            ids = [ids]
            assert isinstance(data, dict), "data should be dict when ids is str"
            data = [data]
        assert len(ids) == len(data), "ids and data should have the same length"
        assert len(set(ids)) == len(ids), "ids should be unique"

        with self.database.begin(write=True) as txn:
            for id_, v in zip(ids, data):
                txn.put(id_.encode("utf-8"), self.serializer.serialize(v))
        return

    def __delitem__(self, ids: str | list[str] | np.ndarray) -> None:
        if isinstance(ids, str):
            ids = [ids]
        assert len(set(ids)) == len(ids), "ids should be unique"

        with self.database.begin(write=True) as txn:
            for idx in ids:
                txn.delete(idx.encode("utf-8"))
        return

    def __len__(self) -> int:
        with self.database.begin() as txn:
            return txn.stat()["entries"]

    def __iter__(self):
        with self.database.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode("utf-8")
        return

    @property
    def fields(self) -> list[str]:
        if len(self) == 0:
            return []
        # This may be incorrect if the schema is not consistent
        with self.database.begin() as txn:
            cursor = txn.cursor()
            for _, data in cursor:
                data = self.serializer.deserialize(data)
                break
        return list(data.keys())
