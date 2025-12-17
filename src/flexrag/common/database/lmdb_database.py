import atexit
import mmap
import os
from collections.abc import Sequence

import lmdb
import numpy as np

from ..logging import LOGGER_MANAGER
from .database_base import RetrieverDatabaseBase
from .serializer import SERIALIZERS, SerializerConfig

logger = LOGGER_MANAGER.get_logger("flexrag.database.lmdb")


class LMDBRetrieverDatabase(RetrieverDatabaseBase):
    """A RetrieverDatabase that uses LMDB as the backend storage format."""

    def __init__(
        self,
        database_path: str,
        map_size: int = 1 << 32,
        serializer="msgpack",
        readonly: bool = False,
        readahead: bool | None = None,
        max_readers: int | None = None,
        writemap: bool | None = None,
        map_async: bool | None = None,
        force_warmup: bool = False,
    ) -> None:
        super().__init__()

        # prepare database path
        self.database_path = database_path
        if not os.path.exists(database_path):
            os.makedirs(database_path)

        # open database
        db_kwargs = {
            "map_size": map_size,
            "readonly": readonly,
        }
        if readahead is not None:
            db_kwargs["readahead"] = readahead
        if max_readers is not None:
            db_kwargs["max_readers"] = max_readers
        if writemap is not None:
            db_kwargs["writemap"] = writemap
        if map_async is not None:
            db_kwargs["map_async"] = map_async
        self.database = lmdb.open(database_path, **db_kwargs)
        atexit.register(self.database.close)

        # prepare serializer
        self.serializer = SERIALIZERS.load(SerializerConfig(serializer))

        # warmup database
        self._warmup(force=force_warmup)
        return

    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        normed_ids = self._normalize_ids(idx)

        with self.database.begin() as txn:
            cursor = txn.cursor()
            items = cursor.getmulti(normed_ids)
        if any(i is None for i in items):
            raise KeyError("Some ids are not found in the database")
        items = [self.serializer.deserialize(i[1]) for i in items]

        if isinstance(idx, str):
            return items[0]
        return items

    def __setitem__(
        self, ids: str | list[str] | np.ndarray, data: dict | list[dict]
    ) -> None:
        normed_ids = self._normalize_ids(ids)
        if isinstance(data, dict):
            data = [data]

        if len(normed_ids) != len(data):
            raise ValueError("ids and data should have the same length")
        if len(set(normed_ids)) != len(normed_ids):
            raise ValueError("ids in the same batch must be unique")

        payloads = [
            (id_, self.serializer.serialize(v)) for id_, v in zip(normed_ids, data)
        ]

        try:
            with self.database.begin(write=True) as txn:
                cursor = txn.cursor()
                cursor.putmulti(payloads)
        except lmdb.MapFullError:
            self._grow_map_size()
            with self.database.begin(write=True) as txn:
                cursor = txn.cursor()
                cursor.putmulti(payloads)
        return

    def __delitem__(self, ids: str | list[str] | np.ndarray) -> None:
        normed_ids = self._normalize_ids(ids)
        if len(set(normed_ids)) != len(normed_ids):
            raise ValueError("ids in the same batch must be unique")

        with self.database.begin(write=True) as txn:
            for id_ in normed_ids:
                txn.delete(id_)
        return

    def __len__(self) -> int:
        return self.database.stat()["entries"]

    def __iter__(self):
        with self.database.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key.decode("utf-8")
        return

    def _normalize_ids(self, ids: str | Sequence[str] | np.ndarray) -> list[bytes]:
        if isinstance(ids, str):
            ids_list = [ids.encode("utf-8")]
        elif isinstance(ids, np.ndarray):
            ids_list = [str(x).encode("utf-8") for x in ids.tolist()]
        elif isinstance(ids, Sequence):
            ids_list = [str(x).encode("utf-8") for x in ids]
        else:
            raise TypeError("ids must be str, Sequence[str], or numpy.ndarray")
        if any(i is None or i == b"" for i in ids_list):
            raise ValueError("ids must be non-empty strings")
        return ids_list

    def _grow_map_size(self, increment: int = 1 << 30) -> None:
        info = self.database.info()
        current_size = info["map_size"]
        new_size = current_size + increment
        self.database.set_mapsize(new_size)
        return

    def _warmup(self, force: bool = False) -> None:
        data_file = os.path.join(self.database_path, "data.mdb")
        if os.path.exists(data_file):
            if hasattr(mmap, "MADV_WILLNEED") and not force:
                with open(data_file, "rb") as f:
                    fd = f.fileno()
                    mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                    mm.madvise(mmap.MADV_WILLNEED)
                    mm.close()
            else:
                with open(data_file, "rb") as f:
                    while f.read(64 * 1024 * 1024):
                        pass
        logger.info("LMDB database warmup completed.")
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

    def close(self) -> None:
        self.database.close()
        atexit.unregister(self.database.close)
        return
