from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class RuntimeCall:
    """A primitive method invocation submitted to a runtime target.

    ``RuntimeCall`` is the scheduler-facing unit of work. Typed handles build
    one or more calls from their public API inputs, while targets decide how
    and where each call executes.

    :param method: Raw resource method name to invoke.
    :param args: Positional arguments passed to the raw method.
    :param kwargs: Keyword arguments passed to the raw method.
    :param weight: Progress accounting weight for this call. It does not affect
        scheduling order or result order.
    :raises ValueError: If ``weight`` is not positive.
    """

    method: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    weight: int = 1

    def __post_init__(self) -> None:
        if self.weight <= 0:
            raise ValueError("RuntimeCall.weight must be greater than 0.")
        return


class RuntimeTarget(Protocol):
    """Execution boundary for typed resource handles.

    A runtime target owns the execution policy for a raw resource: in-process,
    process-isolated, or async-first. Handles call this protocol and never hold
    the raw resource directly. Implementations also own raw resource lifecycle.
    """

    @property
    def batch_size(self) -> int:
        """Return the public-call batch size chosen for this target."""
        ...

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Synchronously invoke one raw resource method.

        :param method: Raw resource method name.
        :param args: Positional arguments for the raw method.
        :param kwargs: Keyword arguments for the raw method.
        :returns: The raw method result.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    async def async_call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously invoke one raw resource method.

        :param method: Raw resource method name.
        :param args: Positional arguments for the raw method.
        :param kwargs: Keyword arguments for the raw method.
        :returns: The raw method result.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    def batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Synchronously execute a batch of primitive runtime calls.

        Results preserve the order of ``calls`` even when the runtime executes
        them concurrently.

        :param calls: Primitive calls to execute.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode forwarded to
            ``SimpleProgressLogger``.
        :param desc: Progress description used for completed call weights.
        :returns: Results in the same order as ``calls``.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    async def async_batch_call(
        self,
        calls: list[RuntimeCall],
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Asynchronously execute a batch of primitive runtime calls.

        :param calls: Primitive calls to execute.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode forwarded to
            ``SimpleProgressLogger``.
        :param desc: Progress description used for completed call weights.
        :returns: Results in the same order as ``calls``.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    def getattr(self, name: str) -> Any:
        """Synchronously read an attribute from the target raw resource.

        :param name: Attribute name.
        :returns: Attribute value.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    async def async_getattr(self, name: str) -> Any:
        """Asynchronously read an attribute from the target raw resource.

        :param name: Attribute name.
        :returns: Attribute value.
        :raises RuntimeError: If the target has been closed.
        """
        ...

    def close(self) -> None:
        """Synchronously close the target and its owned raw resource."""
        ...

    async def async_close(self) -> None:
        """Asynchronously close the target and its owned raw resource."""
        ...
