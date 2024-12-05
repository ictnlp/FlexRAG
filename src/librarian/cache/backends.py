import atexit
import json
import os
import pickle
import shelve
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, MutableMapping

import lmdb
from omegaconf import MISSING

from librarian.utils import Choices


class Serializer(ABC):
    """A simple interface for serializing and deserializing objects."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        return

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        return


class PickleSerializer(Serializer):
    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


class JsonSerializer(Serializer):
    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


class MsgpackSerializer(Serializer):
    def __init__(self) -> None:
        try:
            import msgpack

            self.msgpack = msgpack
        except ImportError:
            raise ImportError("Please install msgpack using `pip install msgpack`.")
        return

    def serialize(self, obj: Any) -> bytes:
        return self.msgpack.packb(obj, use_bin_type=True)

    def deserialize(self, data: bytes) -> Any:
        return self.msgpack.unpackb(data, raw=False)


@dataclass
class LMDBBackendConfig:
    db_path: str = MISSING
    serializer: Choices(["pickle", "json", "msgpack"]) = "pickle"  # type: ignore


class LMDBBackend(MutableMapping):
    def __init__(self, cfg: LMDBBackendConfig) -> None:
        self.db_path = cfg.db_path
        if not os.path.exists(os.path.dirname(cfg.db_path)):
            os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
        self.database = lmdb.open(cfg.db_path)
        atexit.register(self.database.close)
        match cfg.serializer:
            case "pickle":
                self.serializer = PickleSerializer()
            case "json":
                self.serializer = JsonSerializer()
            case "msgpack":
                self.serializer = MsgpackSerializer()
            case _:
                raise ValueError(f"Invalid serializer: {cfg.serializer}")
        return

    def __getitem__(self, key: Any) -> Any:
        with self.database.begin() as txn:
            data = txn.get(self.serializer.serialize(key))
        if data is None:
            raise KeyError(key)
        return self.serializer.deserialize(data)

    def __setitem__(self, key: Any, value: Any) -> None:
        with self.database.begin(write=True) as txn:
            txn.put(self.serializer.serialize(key), self.serializer.serialize(value))
        return

    def __delitem__(self, key: Any) -> None:
        with self.database.begin(write=True) as txn:
            txn.delete(self.serializer.serialize(key))
        return

    def __contains__(self, key: Any) -> bool:
        with self.database.begin() as txn:
            return txn.get(self.serializer.serialize(key)) is not None

    def __len__(self) -> int:
        with self.database.begin() as txn:
            return txn.stat()["entries"]

    def __iter__(self):
        with self.database.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield self.serializer.deserialize(key)
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_path={self.db_path}, len={len(self)})"


@dataclass
class ShelveBackendConfig:
    db_path: str = MISSING
    protocal: int = pickle.DEFAULT_PROTOCOL


class ShelveBackend(MutableMapping):
    def __init__(self, cfg: ShelveBackendConfig) -> None:
        if not os.path.exists(cfg.db_path):
            os.makedirs(cfg.db_path, exist_ok=True)
        self.db_path = os.path.join(cfg.db_path, "cache")
        self.protocal = cfg.protocal
        return

    def __getitem__(self, key: Any) -> Any:
        key_ = json.dumps(key)
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            return database[key_]

    def __setitem__(self, key: Any, value: Any) -> None:
        key_ = json.dumps(key)
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            database[key_] = value
        return

    def __delitem__(self, key: Any) -> None:
        key_ = json.dumps(key)
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            del database[key_]
        return

    def __contains__(self, key: Any) -> bool:
        key_ = json.dumps(key)
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            return key_ in database

    def __len__(self) -> int:
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            return len(database)

    def __iter__(self):
        with shelve.open(self.db_path, protocol=self.protocal) as database:
            for key in database:
                yield json.loads(key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_path={self.db_path}, len={len(self)})"


@dataclass
class PersistentBackendConfig:
    backend: str = "lmdb"
    lmdb_config: LMDBBackendConfig = field(default_factory=LMDBBackendConfig)
    shelve_config: ShelveBackendConfig = field(default_factory=ShelveBackendConfig)


def load_backend(cfg: PersistentBackendConfig) -> MutableMapping:
    match cfg.backend:
        case "lmdb":
            return LMDBBackend(cfg.lmdb_config)
        case "shelve":
            return ShelveBackend(cfg.shelve_config)
        case "dict":
            return dict()
        case _:
            raise ValueError(f"Invalid backend: {cfg.backend}")
    return
