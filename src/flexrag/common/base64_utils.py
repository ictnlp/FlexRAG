from __future__ import annotations

import base64
import io
from os import PathLike
from typing import Union

import requests
from PIL import Image

ImageLike = Union[Image.Image, str, bytes, bytearray, PathLike]


def image_to_base64(
    image: Image.Image,
    format: str = "PNG",
) -> str:
    """Convert a PIL.Image to base64 (no header, utf-8 string)."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_base64(path: Union[str, PathLike]) -> str:
    """Read an image file from disk and return base64 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def binary_to_base64(data: Union[bytes, bytearray]) -> str:
    """Convert binary data to base64 string."""
    return base64.b64encode(bytes(data)).decode("utf-8")


def url_to_base64(url: str) -> str:
    """Fetch content from URL and return base64 string.

    This is typically used for image URLs, but works for any
    binary response body. Raises `RuntimeError` if `requests`
    is not available and `ValueError` for non-200 responses.
    """

    if requests is None:  # pragma: no cover - environment dependent
        raise RuntimeError("`requests` package is required for url_to_base64")

    resp = requests.get(url, stream=True)
    if not resp.ok:
        raise ValueError(f"failed to fetch url: {url} (status {resp.status_code})")

    return binary_to_base64(resp.content)


def base64_to_binary(b64: str) -> bytes:
    """Convert base64 string back to raw bytes."""
    return base64.b64decode(b64.encode("utf-8"))


def base64_to_image(b64: str) -> Image.Image:
    """Convert base64 string to PIL.Image."""
    data = base64_to_binary(b64)
    return Image.open(io.BytesIO(data))


def base64_to_file(b64: str, path: Union[str, PathLike]) -> None:
    """Decode base64 string and write to file."""
    data = base64_to_binary(b64)
    with open(path, "wb") as f:
        f.write(data)


def binary_to_image(data: Union[bytes, bytearray]) -> Image.Image:
    """Convert binary data (image bytes) to PIL.Image."""
    return Image.open(io.BytesIO(bytes(data)))


def image_to_binary(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL.Image to binary in given format."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return buffer.getvalue()
