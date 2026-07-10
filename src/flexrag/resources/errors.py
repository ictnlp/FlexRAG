from __future__ import annotations

import traceback
from typing import Any


class RuntimeCallError(RuntimeError):
    """Error raised when a runtime call fails across a process boundary."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        runtime_traceback: str | None = None,
    ) -> None:
        """Create a runtime call error.

        :param message: User-facing error message.
        :param error_type: Original runtime exception type name when available.
        :param runtime_traceback: Serialized traceback from the runtime boundary.
        """
        super().__init__(message)
        self.error_type = error_type
        self.runtime_traceback = runtime_traceback
        return


def serialize_error(exc: BaseException) -> dict[str, Any]:
    """Serialize an exception for transport across a runtime boundary."""
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }


def raise_runtime_error(payload: dict[str, Any]) -> None:
    """Raise ``RuntimeCallError`` from a serialized runtime error payload."""
    error_type = payload.get("error_type", "Exception")
    message = payload.get("error", "")
    raise RuntimeCallError(
        f"{error_type}: {message}",
        error_type=error_type,
        runtime_traceback=payload.get("traceback"),
    )
