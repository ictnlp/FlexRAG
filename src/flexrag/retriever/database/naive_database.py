from collections import Counter
from typing import Iterable

import numpy as np

from .databsae_base import RetrieverDatabaseBase


class NaiveRetrieverDatabase(RetrieverDatabaseBase):
    """NaiveRetrieverDatabase is a naive implementation of RetrieverDatabaseBase.
    It stores the data in memory"""

    def __init__(self) -> None:
        super().__init__()
        self.data: dict[str, dict] = {}
        self._fields = Counter()
        return

    def add(self, ids: list[str] | str, data: list[dict] | dict) -> None:
        """Add a batch of data to the database.

        :param data: The data to add to the database.
        :type data: list[dict] | dict
        :param ids: The IDs of the data to add to the database.
        :type ids: list[str] | str
        :raises AssertionError: If the IDs are not unique or empty.
        :raises AssertionError: If the ID already exists.
        :return: None
        :rtype: None
        """
        if isinstance(data, dict):
            assert isinstance(ids, str), "ids should be str when data is dict"
            ids = [ids]
            data = [data]
        assert len(data) == len(ids), "data and ids should have the same length"
        assert len(set(ids)) == len(ids), "ids should be unique"
        for idx, item in zip(ids, data):
            self[idx] = item
        return

    def remove(self, ids: list[str] | str) -> None:
        """Remove a batch of data from the database.

        :param ids: The IDs of the data to remove from the database.
        :type ids: list[str] | str
        :raises AssertionError: If the IDs are not unique or empty.
        :raises AssertionError: If the ID does not exist.
        :return: None
        :rtype: None
        """
        if isinstance(ids, str):
            ids = [ids]
        assert len(set(ids)) == len(ids), "ids should be unique"
        for idx in ids:
            assert idx in self.data, f"ID {idx} does not exist"
            del self[idx]
        return

    def __len__(self) -> int:
        """Get the number of items in the database.

        :return: The number of items in the database.
        :rtype: int
        """
        return len(self.data)

    def __iter__(self) -> Iterable[str]:
        """Get an iterator over the database.

        :return: An iterator over the database.
        :rtype: Iterable[dict]
        """
        return iter(self.data.keys())

    def __getitem__(self, idx: str | list[str] | np.ndarray) -> dict | list[dict]:
        """
        Get an / a batch of data from the database.

        :param idx: The index of the data to get.
        :type idx: str | list[str] | np.ndarray
        :return: The data from the database.
        :rtype: dict | list[dict]
        :raises TypeError: If the index is not str, list[str], np.ndarray.
        """
        if isinstance(idx, str):
            return self.data[idx]
        elif isinstance(idx, (list, tuple, np.ndarray)):
            return [self.data[i] for i in idx]
        raise TypeError(
            f"Index should be str, list[str], or np.ndarray, but got {type(idx)}"
        )

    def __setitem__(self, idx: str, item: dict) -> None:
        """
        Set an item in the database.

        :param idx: The index of the data to set.
        :type idx: str
        :param item: The data to set.
        :type item: dict
        :return: None
        :rtype: None
        """
        self.data[idx] = item
        self._fields.update(item.keys())
        return

    def __delitem__(self, idx: str) -> None:
        """
        Delete an item from the database.

        :param idx: The index of the data to delete.
        :type idx: str
        :return: None
        :rtype: None
        """
        self._fields.subtract(self.data[idx].keys())
        del self.data[idx]
        return

    @property
    def fields(self) -> list[str]:
        """
        Get the fields of the database.

        :return: The fields of the database.
        :rtype: list[str]
        """
        return [field for field, count in self._fields.items() if count > 0]
