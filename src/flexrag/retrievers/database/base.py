from abc import abstractmethod
from typing import Iterable, MutableMapping, overload

import numpy as np


class RetrieverDatabaseBase(MutableMapping[str, dict]):
    """Abstract key-value store for retriever contexts."""

    def set(self, ids: list[str] | str, data: list[dict] | dict) -> None:
        """Add one or more context payloads to the database.

        :param ids: Context IDs to add.
        :param data: Context payloads to store.
        :return: None.
        """
        return self.__setitem__(ids, data)

    def remove(self, ids: list[str] | str | np.ndarray) -> None:
        """Remove one or more contexts from the database.

        :param ids: Context IDs to remove.
        :return: None.
        """
        return self.__delitem__(ids)

    @overload
    def __getitem__(self, idx: str) -> dict:
        """Get one context payload by ID.

        :param idx: Context ID.
        :return: Context payload.
        """
        return

    @overload
    def __getitem__(self, idx: list[str] | np.ndarray) -> list[dict]:
        """Get multiple context payloads by ID.

        :param idx: Context IDs.
        :return: Context payloads.
        """
        return

    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        """Get one or more context payloads by ID.

        :param idx: Context ID or context IDs.
        :return: Context payload or payloads.
        """
        return self.get(idx)

    @abstractmethod
    def __setitem__(
        self, idx: str | list[str] | np.ndarray, data: dict | list[dict]
    ) -> None:
        """Set one or more context payloads by ID.

        :param idx: Context ID or context IDs.
        :param data: Context payload or payloads.
        :return: None.
        """
        return

    @abstractmethod
    def __delitem__(self, ids: str | list[str] | np.ndarray) -> None:
        """Delete one or more context payloads by ID.

        :param ids: Context ID or context IDs.
        :return: None.
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """Return fields stored by the database.

        :return: Field names.
        """
        return

    @property
    def ids(self) -> Iterable[str]:
        """Return an iterable over context IDs.

        :return: Context IDs.
        """
        return self.keys()

