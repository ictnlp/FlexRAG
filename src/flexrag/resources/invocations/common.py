from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def split_batches(items: Sequence[T], batch_size: int) -> list[list[T]]:
    """Split items into fixed-size batches.

    :param items: Items to split.
    :param batch_size: Maximum batch size.
    :return: Batches preserving input order.
    :raises ValueError: If ``batch_size`` is not greater than zero.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    return [list(items[i : i + batch_size]) for i in range(0, len(items), batch_size)]


def unwrap_exception_group(exc: Exception) -> Exception:
    """Unwrap single-child exception groups produced by task groups."""
    while isinstance(exc, ExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc
