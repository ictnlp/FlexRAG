from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def _iter_batches(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    iterator = iter(items)
    return iter(lambda: list(islice(iterator, batch_size)), [])
