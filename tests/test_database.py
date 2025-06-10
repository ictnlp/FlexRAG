import tempfile
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from flexrag.database import (
    LMDBRetrieverDatabase,
    NaiveRetrieverDatabase,
    RetrieverDatabaseBase,
)
from flexrag.database.serializer import (
    JsonSerializer,
    MsgpackSerializer,
    PickleSerializer,
    SerializerBase,
)
from flexrag.datasets import LineDelimitedDataset, LineDelimitedDatasetConfig


class TestDatabase:

    def run_basic_operations(self, database: RetrieverDatabaseBase):
        # test `__set__` operation
        database["key1"] = {"data": "value1"}
        database["key2"] = {"data": "value2"}
        database["key3"] = {"data": "value3"}

        # test `get` operation
        assert database.get("key1") == {"data": "value1"}
        assert database.get("key2") == {"data": "value2"}
        assert database.get("key3") == {"data": "value3"}

        # test `__get__` operation
        assert database["key1"] == {"data": "value1"}
        assert database["key2"] == {"data": "value2"}
        assert database["key3"] == {"data": "value3"}

        # test `__len__` operation
        assert len(database) == 3

        # test `__contains__` operation
        assert "key1" in database
        assert "key2" in database
        assert "key3" in database
        assert "key4" not in database
        assert "key5" not in database
        assert "key6" not in database

        # test popitem & pop
        key, value = database.popitem()
        assert len(database) == 2
        assert isinstance(key, str)
        assert isinstance(value, dict)
        if "key2" in database:
            value = database.pop("key2")
            assert len(database) == 1
            assert value == {"data": "value2"}
        else:
            value = database.pop("key1")
            assert len(database) == 1
            assert value == {"data": "value1"}

        # test clear & del
        database.clear()
        database["key3"] = {"data": "value3"}
        database["key4"] = {"data": "value4"}
        del database["key3"]
        assert len(database) == 1
        database.clear()
        assert len(database) == 0

        # test update & setdefault
        database.update({"key1": {"data": "value1"}, "key2": {"data": "value2"}})
        assert len(database) == 2
        assert database.setdefault("key1") == {"data": "value1"}
        assert database.setdefault("key3", {"data": "value3"}) == {"data": "value3"}
        assert len(database) == 3

        # test __iter__ & keys & values & items
        assert set(database) == {"key1", "key2", "key3"}
        assert set(database.keys()) == {"key1", "key2", "key3"}
        expected_values = [
            {"data": "value1"},
            {"data": "value2"},
            {"data": "value3"},
        ]
        actual_values = database.values()
        for actual_value in actual_values:
            assert actual_value in expected_values
            expected_values.remove(actual_value)
        assert len(expected_values) == 0
        expected_items = [
            ("key1", {"data": "value1"}),
            ("key2", {"data": "value2"}),
            ("key3", {"data": "value3"}),
        ]
        actual_items = database.items()
        for actual_item in actual_items:
            assert actual_item in expected_items
            expected_items.remove(actual_item)
        assert len(expected_items) == 0

        # test __eq__ & __ne__
        dict1 = {
            "key1": {"data": "value1"},
            "key2": {"data": "value2"},
            "key4": {"data": "value4"},
        }
        dict2 = {
            "key1": {"data": "value1"},
            "key2": {"data": "value2"},
            "key3": {"data": "value3"},
        }
        assert database != dict1
        assert database == dict2
        return

    def run_batched_operations(self, database: RetrieverDatabaseBase):
        corpus_path = str(Path(__file__).parent / "testcorp" / "testcorp.jsonl")
        dataset1 = [
            i
            for i in LineDelimitedDataset(
                LineDelimitedDatasetConfig(
                    file_paths=[corpus_path],
                    data_ranges=[[0, 10000]],
                )
            )
        ]
        dataset2 = [
            i
            for i in LineDelimitedDataset(
                LineDelimitedDatasetConfig(
                    file_paths=[corpus_path],
                    data_ranges=[[10000, 20000]],
                )
            )
        ]
        indices1 = [str(i) for i in range(10000)]
        indices2 = [str(i) for i in range(10000, 20000)]

        # test `set` method
        database.set(indices1, dataset1)
        assert len(database) == 10000

        # test `get` method
        data = database.get(indices1[:3])
        assert len(data) == 3
        assert data[0]["text"] == dataset1[0]["text"]
        assert data[1]["text"] == dataset1[1]["text"]
        assert data[2]["text"] == dataset1[2]["text"]
        data = database.get(indices1[3:6])
        assert len(data) == 3
        assert data[0]["text"] == dataset1[3]["text"]
        assert data[1]["text"] == dataset1[4]["text"]
        assert data[2]["text"] == dataset1[5]["text"]
        assert isinstance(database[indices1[0]], dict)
        assert isinstance(database[[indices1[0]]], list)

        # test `get` method with non-existing ids
        data = database.get(["1651651231", "15611516"], default=None)
        assert data is None

        # test `__setitem__` method
        database[indices2] = dataset2
        assert len(database) == 20000

        # test `__getitem__` method
        data = database[["10000", "10001", "10002"]]
        assert len(data) == 3
        assert data[0]["text"] == dataset2[0]["text"]
        assert data[1]["text"] == dataset2[1]["text"]
        assert data[2]["text"] == dataset2[2]["text"]
        assert set(database.ids) == set(indices1 + indices2)

        # test `remove` method
        database.remove(indices1)
        assert len(database) == 10000

        # test `__delitem__` method
        del database[indices2[:5000]]
        assert len(database) == 5000

        # test `__contains__` method
        assert indices1[1000] not in database
        assert indices2[0] not in database
        assert indices2[5000] in database

        # test `__iter__` method
        ids = []
        for idx in database:
            assert idx in indices2[5000:]
            ids.append(idx)
        assert len(ids) == 5000

        # test `fields` method
        assert database.fields == list(dataset1[0].keys())

        # test `clear` method
        database.clear()
        assert len(database) == 0

    def test_lmdb_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            database = LMDBRetrieverDatabase(tmpdir)
            self.run_basic_operations(database)
        with tempfile.TemporaryDirectory() as tmpdir:
            database = LMDBRetrieverDatabase(tmpdir)
            self.run_batched_operations(database)
        return

    def test_dict_database(self):
        database = NaiveRetrieverDatabase()
        self.run_basic_operations(database)
        database = NaiveRetrieverDatabase()
        self.run_batched_operations(database)
        return


class TestSerializer:
    data_units = {
        "bool": [
            True,
            False,
        ],
        "int": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            -1,
            2147483648,
            -2147483648,
        ],
        "float": [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            -1.0,
            1 << 32,
            -1 << 32,
        ],
        "str": [
            "",
            "a",
            "a quick brown fox jumps over a lazy dog.",
            "你好，中国。",
            "Test String" * 1000,
        ],
        "list": [
            [0],
            [[0]],
            [1, 2, 3],
            [[1.0, 2.0], 2.0, 3.0],
            [1, 2.0, ["3"]],
            [1, "2", 3],
        ],
        "dict": [
            {
                "data_1": "data_1",
                "data_2": "data_2",
            },
            {
                "data_1": "data_1",
                "data_2": 2.0,
            },
            {
                "data_1": "data_1",
                "data_2": {"sub_data_1": [1, 2.0, "3"], "sub_data_2": {}},
            },
        ],
        "set": [
            {1, 2, 3},
            set(),
            {"x", "y", "z"},
        ],
        "tuple": [
            (0,),
            ((0,),),
            (1, 2, 3),
            ((1.0, 2.0), 2.0, 3.0),
            (1, 2.0, ("3",)),
            (1, "2", 3),
        ],
        "np.generic": [
            np.int32(0),
            np.int64(1),
            np.float16(2.0),
            np.float32(3.0),
            np.float64(4.0),
            np.bool_(True),
            np.bool_(False),
            np.str_("Test String"),
            np.bytes_("Test Bytes"),
            np.complex64(1 + 2j),
        ],
        "np.ndarray": [
            np.array([1, 2, 3], dtype=np.int32),
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ],
        "Image.Image": [
            Image.new("RGB", (100, 100), color=(255, 0, 0)),
            Image.new("L", (100, 100), color=0),
            Image.new("RGBA", (100, 100), color=(255, 0, 0, 255)),
            Image.new("CMYK", (100, 100), color=(255, 0, 0, 255)),
            Image.new("1", (100, 100), color=1),
            Image.new("I", (100, 100), color=1),
            Image.new("F", (100, 100), color=1.0),
            Image.new("P", (100, 100), color=1),
            Image.new("LA", (100, 100), color=(255, 0)),
        ],
        "omegaconf": [
            OmegaConf.create({"key": "value"}),
            OmegaConf.create({"key": 1}),
            OmegaConf.create({"key": 1.0}),
            OmegaConf.create({"key": [1, 2, 3]}),
            OmegaConf.create({"key": {"sub_key": "sub_value"}}),
            OmegaConf.create([{"key": "value"}, {"key": 1}]),
            OmegaConf.create([{"key": 1.0}, {"key": [1, 2, 3]}]),
        ],
    }

    @staticmethod
    def is_equal(a, b):
        if isinstance(a, (str, int, float, bool)):
            return a == b
        elif isinstance(a, (list, dict, set, tuple)):
            return a == b
        elif isinstance(a, (np.ndarray, np.generic)):
            return np.array_equal(a, b)
        elif isinstance(a, Image.Image):
            return a.tobytes() == b.tobytes()
        elif isinstance(a, (DictConfig, ListConfig)):
            return OmegaConf.to_container(a, resolve=True) == OmegaConf.to_container(
                b, resolve=True
            )
        elif isinstance(a, ListConfig):
            return OmegaConf.to_container(a, resolve=True) == OmegaConf.to_container(
                b, resolve=True
            )
        else:
            raise ValueError(f"Unsupported data type: {type(a)}")

    def run_test(self, serializer: SerializerBase):

        def run_specific_type(data_units: list):
            for data_unit in data_units:
                bin = serializer.serialize(data_unit)
                data = serializer.deserialize(bin)
                assert TestSerializer.is_equal(
                    data_unit, data
                ), f"Data: {data_unit}, Deserialized: {data}"
            return

        if serializer.allowed_types is not None:
            allowed_types = serializer.allowed_types
        else:
            allowed_types = self.data_units.keys()

        for data_type in allowed_types:
            assert data_type in self.data_units, f"Unsupported data type: {data_type}"
            run_specific_type(self.data_units[data_type])
        return

    def test_json_serializer(self):
        serializer = JsonSerializer()
        self.run_test(serializer)
        return

    def test_pickle_serializer(self):
        serializer = PickleSerializer()
        self.run_test(serializer)
        return

    def test_msgpack_serializer(self):
        serializer = MsgpackSerializer()
        self.run_test(serializer)
        return
