import os
import shutil
from collections import OrderedDict
from collections.abc import ItemsView, Iterable, KeysView, ValuesView
from functools import cached_property
from hashlib import sha1

import lance
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from flexrag.utils import LOGGER_MANAGER

from .databsae_base import RetrieverDatabaseBase

logger = LOGGER_MANAGER.get_logger("lance_database")


class LanceValuesView(ValuesView):
    """A helper class for faster iterating over the lance database."""

    def __init__(self, mapping: "LanceRetrieverDatabase"):
        super().__init__(mapping)
        self._mapping = mapping
        return

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        if self._mapping.database is None:
            return iter([])
        for batch in self._mapping.database.to_batches():
            for item in batch.to_pandas().to_dict(orient="records"):
                item.pop(self._mapping._id_field_name)
                yield item


class LanceItemsView(ItemsView):
    """A helper class for faster iterating over the lance database."""

    def __init__(self, mapping: "LanceRetrieverDatabase"):
        super().__init__(mapping)
        self._mapping = mapping
        return

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        if self._mapping.database is None:
            return iter([])
        for batch in self._mapping.database.to_batches():
            for item in batch.to_pandas().to_dict(orient="records"):
                key = item.pop(self._mapping._id_field_name)
                yield key, item


class LanceKeysView(KeysView):
    """A helper class for faster iterating over the lance database."""

    def __init__(self, mapping: "LanceRetrieverDatabase"):
        super().__init__(mapping)
        self._mapping = mapping
        return

    def __len__(self):
        return len(self._mapping)

    def __iter__(self):
        return self._mapping._ids.__iter__()

    def __contains__(self, key):
        return self._mapping._ids.__contains__(key)


class LanceRetrieverDatabase(RetrieverDatabaseBase):
    """LanceRetrieverDatabase is a class that implements the RetrieverDatabaseBase interface.
    It uses the Lance database to store the data.
    This dataset is experimental and may not be stable.
    """

    def __init__(self, database_path: str) -> None:
        super().__init__()
        self.database_path = database_path
        self._ids = OrderedDict()  # cached for faster visit
        if os.path.exists(database_path):
            self.database = lance.dataset(database_path)
            for batch in self.database.to_batches():
                for item in batch.to_pandas().to_dict(orient="records"):
                    self._ids[item[self._id_field_name]] = None
        else:
            self.database = None
        return

    def add(self, ids: list[str] | str, data: list[dict] | dict) -> None:
        """Add a batch of data to the database.

        :param ids: The IDs of the data to add to the database.
        :type ids: list[str] | str
        :param data: The data to add to the database.
        :type data: list[dict] | dict
        :raises AssertionError: If the IDs are not unique or empty.
        :raises AssertionError: If the ID already exists.
        :return: None
        :rtype: None
        """
        # check the arguments
        if isinstance(data, dict):
            assert isinstance(ids, str), "ids should be str when data is dict"
            ids = [ids]
            data = [data]
        assert len(data) == len(ids), "data and ids should have the same length"
        assert len(set(ids)) == len(ids), "ids should be unique"

        # remove exist ids
        exist_ids = [idx for idx in ids if idx in self._ids]
        if len(exist_ids) > 0:
            self.remove(exist_ids)

        # add the data to the database
        data = pd.DataFrame(data)
        data[self._id_field_name] = ids
        if self.database is None:
            self.database = lance.write_dataset(
                data,
                uri=self.database_path,
                mode="create",
            )
        else:
            self.database.insert(data, mode="append")

        # update ids cache
        for idx in ids:
            self._ids[idx] = None
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
        assert len(ids) > 0, "ids should not be empty"
        assert self.database is not None, "database is empty"
        assert self.database.count_rows() > 0, "database is empty"

        # remove the data from the database
        # as lance has issue with deleting data using pa.compute.Expression
        # so we use str to filter the data
        ids_ = [f"'{i}'" for i in ids]
        filter_str = f"{self._id_field_name} in ({', '.join([i for i in ids_])})"
        self.database.delete(filter_str)

        # update the _ids cache
        for idx in ids:
            self._ids.pop(idx)
        return

    def __getitem__(self, ids: str | list[str] | np.ndarray) -> list[dict]:
        """Get (a batch of) data from the database.

        :param ids: The ids of the data to get.
        :type ids: str | list[str] | np.ndarray
        :raises AssertionError: If the type of ids is not list or np.ndarray.
        :return: The data from the database.
        :rtype: list[dict]
        """
        if self.database is None:
            raise KeyError("database is empty")
        if isinstance(ids, np.ndarray):
            ids_ = ids.tolist()
        elif isinstance(ids, str):
            ids_ = [ids]
        else:
            assert isinstance(ids, list), "ids should be str, list or np.ndarray"
            ids_ = ids
        assert isinstance(ids_, list), "indices should be list or np.ndarray"
        assert all([isinstance(i, str) for i in ids_]), "ids should be str"

        try:
            data = self.database.to_table(
                filter=ds.field(self._id_field_name).isin(ids_)
            ).to_pandas()
            data.drop(columns=[self._id_field_name], inplace=True)
        except:
            raise KeyError

        if len(data) == 0:
            raise KeyError(f"ids {ids} not found in the database")
        data = data.to_dict(orient="records")

        if isinstance(ids, str):
            return data[0]
        return data

    def __setitem__(self, ids: str, data: dict) -> None:
        """
        Set an item in the database.

        :param idx: The index of the data to set.
        :type idx: str
        :param item: The data to set.
        :type item: dict
        :return: None
        :rtype: None
        """
        self.add(ids, data)
        return

    def __delitem__(self, ids: str) -> None:
        """
        Delete an item from the database.

        :param idx: The index of the data to delete.
        :type idx: str
        :return: None
        :rtype: None
        """
        self.remove(ids)
        return

    def clear(self) -> None:
        """Clear the database.

        :return: None
        :rtype: None
        """
        if self.database is None:
            return
        self.database.drop(self.database_path)
        self.database = None
        self._ids = OrderedDict()
        return

    def __len__(self) -> int:
        """Get the number of items in the database.

        :return: The number of items in the database.
        :rtype: int
        """
        if self.database is None:
            return 0
        return self.database.count_rows()

    def __iter__(self) -> Iterable[str]:
        """Get an iterator over the database.

        :return: An iterator over the database.
        :rtype: Iterable[str]
        """
        if self.database is None:
            return iter([])
        for batch in self.database.to_batches():
            for ids in batch.to_pandas()[self._id_field_name]:
                yield ids
        return

    def values(self) -> ValuesView:
        return LanceValuesView(self)

    def items(self) -> ItemsView:
        return LanceItemsView(self)

    def keys(self) -> KeysView:
        return LanceKeysView(self)

    @property
    def fields(self) -> list[str]:
        """Get the fields of the database.

        :return: The fields of the database.
        :rtype: list[str]
        """
        if self.database is None:
            return []
        fields = self.database.schema.names
        fields.remove(self._id_field_name)
        return fields

    def defragment(self) -> None:
        """Clear up the database."""
        if self.database is None:
            return
        if len(self.database.versions()) <= 1:
            return
        # compact the database
        # we find that the `compact_files` operation will change the order of the rows
        # so we compact the files directly by reading and writing the data
        logger.info("Compacting the database. This may take a while.")
        parent_path = os.path.dirname(self.database_path)
        new_data_path = os.path.join(parent_path, "tmp.lance")
        ori_data_path = os.path.join(parent_path, "database.lance")
        lance.write_dataset(self.database, uri=new_data_path, mode="create")
        shutil.rmtree(ori_data_path)
        shutil.move(new_data_path, ori_data_path)
        self.database = lance.dataset(ori_data_path)
        return

    @cached_property
    def _id_field_name(self) -> str:
        """Get the ID field name."""
        return sha1("context_id".encode()).hexdigest()
