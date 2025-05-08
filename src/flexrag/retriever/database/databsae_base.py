from abc import abstractmethod
from typing import Iterable, MutableMapping, overload

import numpy as np


class RetrieverDatabaseBase(MutableMapping[str, dict]):
    """RetrieverDatabaseBase is an abstract base class for a retriever database.
    It provides an interface for adding, and getting data from the database.
    The database should act as a key-value store, where the key is a unique ID and the value is a dictionary of data.

    The subclasses should implement the following methods:
        * __setitem__: Add (a batch of) data to the database.
        * __delitem__: Remove (a batch of) data from the database.
        * __getitem__: Get (a batch of) data from the database.
        * __iter__: Iterate over the keys in the database.
        * __len__: Get the number of items in the database.
        * fields: Get the fields of the database.
        * ids: Get the IDs of the items in the database.
    """

    def set(self, ids: list[str] | str, data: list[dict] | dict) -> None:
        """Add (a batch of) data to the database.

        :param data: The data to add to the database.
        :type data: list[dict] | dict
        :param ids: The IDs of the data to add to the database.
        :type ids: list[str] | str
        :return: None
        :rtype: None
        """
        return self.__setitem__(ids, data)

    def remove(self, ids: list[str] | str | np.ndarray) -> None:
        """Remove (a batch of) data from the database.

        :param ids: The IDs of the data to remove from the database.
        :type ids: list[str] | str | np.ndarray
        :return: None
        :rtype: None
        """
        return self.__delitem__(ids)

    @overload
    def __getitem__(self, idx: str) -> dict:
        """
        Get an item from the database.

        param: idx: The index of the item to get.
        type: idx: str
        return: The item from the database.
        rtype: dict
        """
        return

    @overload
    def __getitem__(self, idx: list[str] | np.ndarray) -> list[dict]:
        """
        Get a batch of items from the database.

        param: idx: The index of the items to get.
        type: idx: list[str] | np.ndarray
        return: The items from the database.
        rtype: list[dict]
        """
        return

    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        """
        Get (a batch of) item from the database.

        param: idx: The index of the item to get.
        type: idx: str | list[str] | np.ndarray
        return: The item from the database.
        rtype: dict | list[dict]
        """
        return self.get(idx)

    @abstractmethod
    def __setitem__(
        self, idx: str | list[str] | np.ndarray, data: dict | list[dict]
    ) -> None:
        """
        Set (a batch of) item in the database.

        params: idx: The index of the item to set.
        type: idx: str | list[str] | np.ndarray
        params: data: The data to set.
        type: data: dict | list[dict]
        return: None
        rtype: None
        """
        return

    @abstractmethod
    def __delitem__(self, ids: str | list[str] | np.ndarray) -> None:
        """
        Delete (a batch of) item from the database.
        params: ids: The index of the item to delete.
        type: ids: str | list[str] | np.ndarray
        return: None
        rtype: None
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """
        Get the fields of the database.

        Returns:
            list[str]: The fields of the database.
        """
        return

    @property
    def ids(self) -> Iterable[str]:
        """
        Get the IDs of the items in the database.

        Returns:
            Iterable[str]: An iterable of IDs in the database.
        """
        return self.keys()
