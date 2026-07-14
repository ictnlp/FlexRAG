import json
import pickle
from dataclasses import asdict, is_dataclass
from typing import Any, Literal, Protocol, overload

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

from .configure import Register


class SerializerProtocol(Protocol):
    """Structural interface for serializers registered with FlexRAG."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize an object into bytes.

        :param obj: Object to serialize.
        :return: Serialized bytes.
        """
        ...

    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes into an object.

        :param data: Serialized bytes.
        :return: Deserialized object.
        """
        ...

    @property
    def allowed_types(self) -> list[str] | None:
        """Return the type names supported by this serializer.

        ``None`` means the serializer accepts most Python objects.

        :return: Supported type names or ``None``.
        """
        ...


SERIALIZERS = Register[SerializerProtocol]("serializer")


@SERIALIZERS("pickle")
class PickleSerializer:
    """Serializer based on :mod:`pickle`.

    Pickle supports most Python objects, but it is unsafe for untrusted data.
    """

    def serialize(self, obj: Any) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes) -> Any:
        return pickle.loads(data)

    @property
    def allowed_types(self) -> list[str] | None:
        """Return ``None`` because pickle supports most Python objects."""
        return None


@SERIALIZERS("json")
class JsonSerializer:
    """Serializer based on :mod:`json` with FlexRAG convenience extensions."""

    def __init__(self) -> None:
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (DictConfig, ListConfig)):
                    return OmegaConf.to_container(obj, resolve=True)
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if is_dataclass(obj):
                    return asdict(obj)
                return super().default(obj)

        self.encoder = CustomEncoder

    @overload
    def serialize(self, obj: Any) -> bytes: ...

    @overload
    def serialize(
        self,
        obj: Any,
        to_bytes: Literal[True],
        ensure_ascii: bool = True,
        indent: int | None = None,
        **kwargs,
    ) -> bytes: ...

    @overload
    def serialize(
        self,
        obj: Any,
        to_bytes: Literal[False],
        ensure_ascii: bool = True,
        indent: int | None = None,
        **kwargs,
    ) -> str: ...

    @overload
    def serialize(
        self,
        obj: Any,
        to_bytes: bool,
        ensure_ascii: bool = True,
        indent: int | None = None,
        **kwargs,
    ) -> bytes | str: ...

    def serialize(
        self,
        obj: Any,
        to_bytes: bool = True,
        ensure_ascii: bool = True,
        indent: int | None = None,
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

    @property
    def allowed_types(self) -> list[str]:
        return ["str", "int", "float", "bool", "dict", "list"]


_JsonSerializer = JsonSerializer()


def json_dump(
    obj: Any,
    to_bytes: bool = True,
    ensure_ascii: bool = True,
    indent: int | None = None,
    **kwargs,
) -> bytes | str:
    """Serialize an object into JSON with FlexRAG convenience extensions.

    :param obj: Object to serialize.
    :param to_bytes: Whether to return UTF-8 encoded bytes. Defaults to
        ``True``.
    :param ensure_ascii: Whether non-ASCII characters should be escaped.
        Defaults to ``True``.
    :param indent: Optional indentation level for pretty printing.
    :param kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.
    :return: JSON bytes or text.
    """
    return _JsonSerializer.serialize(obj, to_bytes, ensure_ascii, indent, **kwargs)


@SERIALIZERS("msgpack")
class MsgpackSerializer:
    """Serializer based on :mod:`msgpack` with additional type support."""

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
            if isinstance(obj, np.ndarray):
                return {
                    "__type__": "np_ndarray",
                    "dtype": obj.dtype.name,
                    "shape": obj.shape,
                    "data": obj.tobytes(),
                }
            if isinstance(obj, np.generic):
                return {
                    "__type__": "np_generic",
                    "dtype": obj.dtype.name,
                    "data": obj.tobytes(),
                }
            if isinstance(obj, Image.Image):
                return {
                    "__type__": "pillow_image",
                    "mode": obj.mode,
                    "size": obj.size,
                    "data": obj.tobytes(),
                }
            if isinstance(obj, (DictConfig, ListConfig)):
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
            if obj["__type__"] == "np_ndarray":
                return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(
                    obj["shape"]
                )
            if obj["__type__"] == "np_generic":
                return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).item()
            if obj["__type__"] == "pillow_image":
                return Image.frombytes(obj["mode"], obj["size"], obj["data"])
            if obj["__type__"] == "omegaconf_config":
                return OmegaConf.create(obj["data"])
            return obj

        return self.msgpack.unpackb(data, raw=False, object_hook=extended_decode)

    @property
    def allowed_types(self) -> list[str]:
        return [
            "str",
            "int",
            "float",
            "bool",
            "list",
            "set",
            "dict",
            "np.ndarray",
            "np.generic",
            "Image.Image",
            "omegaconf",
        ]


SerializerConfig = SERIALIZERS.make_config(default="msgpack")
