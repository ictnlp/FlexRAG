import atexit
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import MutableMapping

import lmdb
from omegaconf import MISSING

from flexrag.utils import Register


class StorageBackendBase(MutableMapping[bytes, bytes]):
    """The Binary Storage Backend For ``PersistentCache``.
    The backend should provide interfaces like ``MutableMapping``.
    Thus, The following methods should be implemented:

        >>> def __getitem__(self, key: bytes) -> bytes:
        ...     pass
        >>> def __setitem__(self, key: bytes, value: bytes) -> None:
        ...     pass
        >>> def __delitem__(self, key: bytes) -> None:
        ...     pass
        >>> def __iter__(self) -> Iterable[bytes]:
        ...     pass
        >>> def __len__(self) -> int:
        ...     pass

    The following methods will be implemented automatically:

        >>> def __contains__(self, key: bytes) -> bool:
        ...     pass
        >>> def keys(self) -> KeysView:
        ...     pass
        >>> def values(self) -> ValuesView:
        ...     pass
        >>> def items(self) -> ItemsView:
        ...     pass
        >>> def get(self, key: bytes, default: Any = None) -> bytes | Any:
        ...     pass
        >>> def __eq__(self, other: StorageBackend) -> bool:
        ...     pass
        >>> def __ne__(self, other: StorageBackend) -> bool:
        ...     pass
        >>> def pop(self, key: bytes, default: Any = None) -> bytes | Any:
        ...     pass
        >>> def popitem(self) -> Tuple:
        ...     pass
        >>> def clear(self) -> None:
        ...     pass
        >>> def update(self, other: MutableMapping) -> None:
        ...     pass
        >>> def setdefault(self, key: bytes, default: Any = None) -> Any:
        ...     pass
    """

    def __repr__(self) -> str:
        f"{self.__class__.__name__}(len={len(self)})"


STORAGEBACKENDS = Register[StorageBackendBase]("storage_backend")


@dataclass
class LMDBBackendConfig:
    db_path: str = MISSING
    db_size: int = 1 << 30  # 2^30 bytes = 1GB


@STORAGEBACKENDS("lmdb", config_class=LMDBBackendConfig)
class LMDBBackend(StorageBackendBase):
    def __init__(self, cfg: LMDBBackendConfig) -> None:
        self.db_path = cfg.db_path
        if not os.path.exists(os.path.dirname(cfg.db_path)):
            os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
        self.database = lmdb.open(cfg.db_path, map_size=cfg.db_size)
        atexit.register(self.database.close)
        return

    def __getitem__(self, key: bytes) -> bytes:
        with self.database.begin() as txn:
            data = txn.get(key)
        if data is None:
            raise KeyError(key)
        return data

    def __setitem__(self, key: bytes, value: bytes) -> None:
        with self.database.begin(write=True) as txn:
            txn.put(key, value)
        return

    def __delitem__(self, key: bytes) -> None:
        with self.database.begin(write=True) as txn:
            txn.delete(key)
        return

    def __len__(self) -> int:
        with self.database.begin() as txn:
            return txn.stat()["entries"]

    def __iter__(self):
        with self.database.begin() as txn:
            cursor = txn.cursor()
            for key, _ in cursor:
                yield key
        return

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_path={self.db_path}, len={len(self)})"


@STORAGEBACKENDS("dict")
class DictBackend(OrderedDict, StorageBackendBase): ...


StorageBackendConfig = STORAGEBACKENDS.make_config(default="dict")
