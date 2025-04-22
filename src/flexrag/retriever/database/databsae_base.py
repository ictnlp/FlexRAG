from abc import abstractmethod
from typing import Iterable, MutableMapping, overload

import numpy as np


class RetrieverDatabaseBase(MutableMapping[str, dict]):
    """RetrieverDatabaseBase is an abstract base class for a retriever database.
    It provides an interface for adding, and getting data from the database.
    The database should act as a key-value store, where the key is a unique ID and the value is a dictionary of data.
    """

    @abstractmethod
    def add(self, ids: list[str], data: list[dict]) -> None:
        """Add a batch of data to the database.

        :param data: The data to add to the database.
        :type data: list[dict] | dict
        :param ids: The IDs of the data to add to the database.
        :type ids: list[str] | str
        :return: None
        :rtype: None
        """
        return

    @abstractmethod
    def remove(self, ids: list[str]) -> None:
        """Remove a batch of data from the database.

        :param ids: The IDs of the data to remove from the database.
        :type ids: list[str] | str
        :return: None
        :rtype: None
        """
        return

    @overload
    def __getitem__(self, idx: str) -> dict:
        """
        Get an item from the database.

        Args:
            index (int): The index of the item to get.

        Returns:
            dict: The item from the database.
        """
        return

    @overload
    def __getitem__(self, idx: list[str] | np.ndarray) -> list[dict]:
        """
        Get an item from the database.

        Args:
            index (list[int] | np.ndarray): The indices of the items to get.

        Returns:
            list[dict]: The items from the database.
        """
        return

    @abstractmethod
    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        """
        Get an item from the database.

        Args:
            index: The index of the item to get.

        Returns:
            dict | list[dict]: The item from the database.
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
