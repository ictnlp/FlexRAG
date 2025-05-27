import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

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
    """A serializer that uses the pickle module.

    PickleSerializer supports almost all python objects.
    However, it is not safe for loading untrusted data.
    """

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)


@SERIALIZERS("json")
class JsonSerializer(SerializerBase):
    """A serializer that uses the json module.

    JsonSerializer supports basic types, including:
    str, int, float, bool, list, dict.
    """

    def __init__(self) -> None:
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, DictConfig):
                    return OmegaConf.to_container(obj, resolve=True)
                if isinstance(obj, np.int64):
                    return int(obj)
                if isinstance(obj, np.int32):
                    return int(obj)
                if isinstance(obj, np.float64):
                    return float(obj)
                if isinstance(obj, np.float32):
                    return float(obj)
                if is_dataclass(obj):
                    return asdict(obj)
                # if hasattr(obj, "to_list"):
                #     return obj.to_list()
                # if hasattr(obj, "to_dict"):
                #     return obj.to_dict()
                return super().default(obj)

        self.encoder = CustomEncoder

    def serialize(
        self,
        obj: Any,
        to_bytes: bool = True,
        ensure_ascii: bool = True,
        indent: int = None,
        **kwargs,
    ) -> bytes | str:
        if to_bytes:
            return json.dumps(obj, cls=self.encoder).encode("utf-8")
        return json.dumps(
            obj,
            cls=self.encoder,
            ensure_ascii=ensure_ascii,
            indent=indent,
            **kwargs,
        )

    def deserialize(self, data: bytes) -> Any:
        return json.loads(data.decode("utf-8"))


_JsonSerializer = JsonSerializer()


def json_dump(
    obj: Any,
    to_bytes: bool = True,
    ensure_ascii: bool = True,
    indent: int = None,
    **kwargs,
) -> bytes | str:
    """A shortcut for serialize the object into JSON format.
    This function extends the json.dumps function to support
    additional types such as DictConfig and ListConfig.

    :param obj: The object to serialize.
    :type obj: Any
    :param to_bytes: Whether to return bytes or str. Defaults to True.
    :type to_bytes: bool, optional
    :param ensure_ascii: Whether to ensure ASCII encoding. Defaults to True.
    :type ensure_ascii: bool, optional
    :param indent: The indentation level for pretty printing. Defaults to None.
    :type indent: int, optional
    :return: The serialized object in JSON format.
    :rtype: bytes | str
    """
    return _JsonSerializer.serialize(obj, to_bytes, ensure_ascii, indent, **kwargs)


@SERIALIZERS("msgpack")
class MsgpackSerializer(SerializerBase):
    """A serializer that uses the msgpack module.

    MsgpackSerializer supports more types than JsonSerializer, including:
    str, int, float, bool, list, set, dict, np.ndarray, np.generic, Image.Image,
    omegaconf.DictConfig, and omegaconf.ListConfig.
    """

    def __init__(self) -> None:
        try:
            import msgpack

            self.msgpack = msgpack
        except ImportError:
            raise ImportError("Please install msgpack using `pip install msgpack`.")
        return

    def serialize(self, obj: Any) -> bytes:
        def extended_encode(obj):
            if isinstance(obj, set):
                return {
                    "__type__": "set",
                    "data": list(obj),
                }
            elif isinstance(obj, np.ndarray):
                return {
                    "__type__": "np_ndarray",
                    "dtype": obj.dtype.name,
                    "shape": obj.shape,
                    "data": obj.tobytes(),
                }
            elif isinstance(obj, np.generic):
                return {
                    "__type__": "np_generic",
                    "dtype": obj.dtype.name,
                    "data": obj.tobytes(),
                }
            elif isinstance(obj, Image.Image):
                return {
                    "__type__": "pillow_image",
                    "mode": obj.mode,
                    "size": obj.size,
                    "data": obj.tobytes(),
                }
            elif isinstance(obj, (DictConfig, ListConfig)):
                return {
                    "__type__": "omegaconf_config",
                    "data": OmegaConf.to_container(obj, resolve=True),
                }
            return obj

        return self.msgpack.packb(obj, use_bin_type=True, default=extended_encode)

    def deserialize(self, data: bytes) -> Any:
        def extended_decode(obj):
            if "__type__" not in obj:
                return obj
            if obj["__type__"] == "set":
                return set(obj["data"])
            elif obj["__type__"] == "np_ndarray":
                return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(
                    obj["shape"]
                )
            elif obj["__type__"] == "np_generic":
                return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).item()
            elif obj["__type__"] == "pillow_image":
                return Image.frombytes(obj["mode"], obj["size"], obj["data"])
            elif obj["__type__"] == "omegaconf_config":
                return OmegaConf.create(obj["data"])
            return obj

        return self.msgpack.unpackb(data, raw=False, object_hook=extended_decode)


SerializerConfig = SERIALIZERS.make_config(default="msgpack")
