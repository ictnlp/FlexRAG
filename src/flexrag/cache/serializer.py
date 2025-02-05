import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

from flexrag.utils import Register


class SerializerBase(ABC):
    """A simple interface for serializing and deserializing python objects."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        return

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        return


SERIALIZERS = Register[SerializerBase]("serializer")


@SERIALIZERS("pickle")
class PickleSerializer(SerializerBase):
    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


@SERIALIZERS("cloudpickle")
class CloudPickleSerializer(SerializerBase):
    def __init__(self):
        try:
            import cloudpickle

            self.pickler = cloudpickle
        except:
            raise ImportError(
                "Please install cloudpickle using `pip install cloudpickle`."
            )
        return

    def serialize(self, obj: Any) -> bytes:
        return self.pickler.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return self.pickler.loads(data)


@SERIALIZERS("json")
class JsonSerializer(SerializerBase):
    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


@SERIALIZERS("msgpack")
class MsgpackSerializer(SerializerBase):
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


SerializerConfig = SERIALIZERS.make_config(default="json")
