import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from flexrag.common.serialization import (
    JsonSerializer,
    MsgpackSerializer,
    PickleSerializer,
    SerializerProtocol,
)


class TestSerialization:
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

    def run_test(self, serializer: SerializerProtocol):

        def run_specific_type(data_units: list):
            for data_unit in data_units:
                bin = serializer.serialize(data_unit)
                data = serializer.deserialize(bin)
                assert TestSerialization.is_equal(data_unit, data), (
                    f"Data: {data_unit}, Deserialized: {data}"
                )
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
