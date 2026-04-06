"""Generic subprocess worker utilities for client-backed local runtimes.

This module provides the lowest-level building blocks used by local process
backends: serializing class references as ``module:qualname`` strings,
reconstructing configs inside worker processes, starting a worker object in a
subprocess, and forwarding attribute/method calls over a simple Pipe-based RPC
protocol.
"""

import asyncio
import importlib
import multiprocessing as mp
import os
import threading
import traceback


def get_symbol_path(obj: object) -> str:
    """Return a stable ``module:qualname`` path for an importable object."""

    return f"{obj.__module__}:{obj.__qualname__}"


def resolve_symbol(path: str):
    """Resolve an object from a ``module:qualname`` path."""

    module_name, qualname = path.split(":", maxsplit=1)
    obj = importlib.import_module(module_name)
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    return obj


def build_worker_config(
    config_cls_path: str,
    config_data: dict,
    visible_device_ids: list[int] | None,
):
    """Rebuild a config object inside a worker process.

    If the config exposes ``device_id``, the worker view is normalized to a
    local device list. After ``CUDA_VISIBLE_DEVICES`` is set in the worker,
    the selected physical GPUs are always exposed as ``[0, 1, ...]``.
    """

    config_cls = resolve_symbol(config_cls_path)
    config = config_cls(**config_data)
    if hasattr(config, "device_id"):
        config.device_id = (
            list(range(len(visible_device_ids))) if visible_device_ids else []
        )
    return config


def _worker_main(
    conn,
    impl_path: str,
    config_cls_path: str,
    config_data: dict,
    visible_device_ids: list[int] | None,
) -> None:
    """Run a worker object in a subprocess and serve simple RPC requests.

    The worker protocol currently supports two request kinds:

    - ``call``: look up an attribute on the worker object and either call it
      with the provided ``args``/``kwargs`` or return the attribute value
      directly;
    - ``close``: stop the request loop and exit the subprocess.
    """

    if not visible_device_ids:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_device_ids))

    worker_obj = resolve_symbol(impl_path)(
        build_worker_config(config_cls_path, config_data, visible_device_ids)
    )
    while True:
        request = conn.recv()
        if request["kind"] == "close":
            break
        if request["kind"] != "call":
            conn.send(
                {
                    "ok": False,
                    "error_type": "ValueError",
                    "error": f"Unsupported worker request kind: {request['kind']}",
                    "traceback": "",
                }
            )
            continue

        try:
            attr = getattr(worker_obj, request["attribute"])
            if callable(attr):
                result = attr(*request["args"], **request["kwargs"])
            else:
                if request["args"] or request["kwargs"]:
                    raise TypeError(
                        f"Attribute {request['attribute']} is not callable and does not accept arguments."
                    )
                result = attr
            conn.send({"ok": True, "data": result})
        except Exception as exc:  # noqa: BLE001
            conn.send(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
    conn.close()
    return


class ProcessWorkerClient:
    """RPC client for a single local worker subprocess.

    A ``ProcessWorkerClient`` owns one spawned subprocess and exposes a small
    attribute-based RPC surface to the parent process. Calls are serialized with
    a lock because each worker has a single Pipe connection and therefore only
    supports one in-flight request at a time.
    """

    def __init__(
        self,
        impl_path: str,
        config_cls_path: str,
        config_data: dict,
        visible_device_ids: list[int] | None,
    ) -> None:
        context = mp.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        self._conn = parent_conn
        self._lock = threading.Lock()
        self._closed = False
        self._process = context.Process(
            target=_worker_main,
            args=(
                child_conn,
                impl_path,
                config_cls_path,
                config_data,
                visible_device_ids,
            ),
            daemon=True,
        )
        self._process.start()
        child_conn.close()
        return

    @property
    def process(self):
        return self._process

    def _call_sync(self, attribute: str, *args, **kwargs):
        """Send a blocking RPC request to the worker and return the result."""

        with self._lock:
            if self._closed:
                raise RuntimeError("Worker has been closed.")
            self._conn.send(
                {
                    "kind": "call",
                    "attribute": attribute,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            response = self._conn.recv()
        if response["ok"]:
            return response["data"]
        raise RuntimeError(
            f"Worker attribute {attribute} failed with "
            f"{response['error_type']}: {response['error']}\n{response['traceback']}"
        )

    async def call(self, attribute: str, *args, **kwargs):
        """Execute a worker RPC without blocking the caller's event loop."""

        return await asyncio.to_thread(self._call_sync, attribute, *args, **kwargs)

    def close(self) -> None:
        """Stop the worker process and clean up the Pipe connection."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.send({"kind": "close"})
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass

        if self._process.is_alive():
            self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
        return
