import json
import pickle
from abc import ABC, abstractmethod
from typing import Any

from flexrag.utils import Register


class SerializerBase(ABC):
    """A simple interface for serializing and deserializing python objects."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize the object into bytes.

        :param obj: The object to serialize.
        :type obj: Any
        :return: The serialized object.
        :rtype: bytes
        """
        return

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize the bytes into an object.

        :param data: The serialized object.
        :type data: bytes
        :return: The deserialized object.
        :rtype: Any
        """
        return


SERIALIZERS = Register[SerializerBase]("serializer")


@SERIALIZERS("pickle")
class PickleSerializer(SerializerBase):
    """A serializer that uses the pickle module."""

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


@SERIALIZERS("cloudpickle")
class CloudPickleSerializer(SerializerBase):
    """A serializer that uses the cloudpickle module."""

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
    """A serializer that uses the json module."""

    def serialize(self, obj: Any) -> bytes:
        return json.dumps(obj).encode("utf-8")

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


@SERIALIZERS("msgpack")
class MsgpackSerializer(SerializerBase):
    """A serializer that uses the msgpack module."""

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


SerializerConfig = SERIALIZERS.make_config(default="pickle")
