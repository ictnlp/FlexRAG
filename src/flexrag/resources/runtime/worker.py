from __future__ import annotations

import asyncio
import inspect
from typing import Any

from ..errors import serialize_error
from ..handles import HANDLE_TYPES, TypedHandle
from ..refs import ResourceRefDescriptor
from ..symbols import resolve_symbol
from .parent_proxy import ParentProxyTarget


def materialize_worker_refs(
    conn: Any,
    refs: dict[str, ResourceRefDescriptor],
) -> dict[str, TypedHandle]:
    """Build worker-side dependency handles around parent proxy targets."""
    materialized = {}
    for param_name, descriptor in refs.items():
        handle_cls = HANDLE_TYPES[descriptor.interface]
        materialized[param_name] = handle_cls(
            ParentProxyTarget(descriptor, conn),
            batching=descriptor.batching,
        )
    return materialized


def worker_main(
    conn: Any,
    raw_cls_path: str,
    config: Any,
    refs: dict[str, ResourceRefDescriptor],
) -> None:
    """Run one process worker that owns a single raw resource instance."""
    try:
        raw_cls = resolve_symbol(raw_cls_path)
        raw = raw_cls(config, **materialize_worker_refs(conn, refs))
    except Exception as exc:  # noqa: BLE001
        response = serialize_error(exc)
        response.update({"kind": "started"})
        conn.send(response)
        conn.close()
        return
    conn.send({"kind": "started", "ok": True})
    while True:
        try:
            request = conn.recv()
        except EOFError:
            break

        kind = request.get("kind")
        if kind == "close":
            _close_raw(raw)
            conn.send({"kind": "closed", "id": request.get("id"), "ok": True})
            break

        if kind != "call":
            conn.send(
                {
                    "kind": "response",
                    "id": request.get("id"),
                    "ok": False,
                    "error_type": "ValueError",
                    "error": f"Unsupported request kind: {kind}",
                    "traceback": "",
                }
            )
            continue

        try:
            method = request["method"]
            if method == "__getattr__":
                result = getattr(raw, request["args"][0])
            else:
                result = getattr(raw, method)(
                    *request.get("args", ()),
                    **request.get("kwargs", {}),
                )
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            conn.send(
                {
                    "kind": "response",
                    "id": request["id"],
                    "ok": True,
                    "data": result,
                }
            )
        except Exception as exc:  # noqa: BLE001
            response = serialize_error(exc)
            response.update({"kind": "response", "id": request.get("id")})
            conn.send(response)
    conn.close()
    return


def _close_raw(raw: Any) -> None:
    """Close a raw resource using ``async_close`` or ``close`` if present."""
    for method_name in ("async_close", "close"):
        close = getattr(raw, method_name, None)
        if not callable(close):
            continue
        result = close()
        if inspect.isawaitable(result):
            asyncio.run(result)
        return
    return
