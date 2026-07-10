from __future__ import annotations

import asyncio
import inspect
import itertools
import multiprocessing as mp
import threading
import time
from typing import TYPE_CHECKING, Any

from ..errors import raise_runtime_error, serialize_error
from ..refs import ResourceRefDescriptor
from ..symbols import symbol_path
from .placement import temporary_env

if TYPE_CHECKING:
    from ..resource_manager import ResourceManager


class ProcessWorkerClient:
    """Client for one worker process that owns one raw resource instance.

    The client sends primitive calls to its worker and pumps dependency requests
    from that worker back into the parent ``ResourceManager`` while waiting for
    a response.
    """

    _ids = itertools.count(1)

    def __init__(
        self,
        raw_cls: type[Any],
        config: Any,
        refs: dict[str, ResourceRefDescriptor],
        manager: ResourceManager,
        env_updates: dict[str, str] | None = None,
    ) -> None:
        """Spawn a worker process.

        :param raw_cls: Raw resource class to instantiate in the worker.
        :param config: Config object sent to the worker constructor.
        :param refs: Ref descriptors materialized as parent-proxy handles.
        :param manager: Parent manager used to serve dependency calls.
        :param env_updates: Environment updates applied while starting the
            worker process.
        """
        from .worker import worker_main

        self._manager = manager
        self._closed = False
        self._lock = threading.Lock()
        context = mp.get_context("spawn")
        parent_conn, child_conn = context.Pipe()
        self._conn = parent_conn
        self._process = context.Process(
            target=worker_main,
            args=(child_conn, symbol_path(raw_cls), config, refs),
            daemon=True,
        )
        with temporary_env(env_updates):
            self._process.start()
        child_conn.close()
        self._wait_for_startup()
        return

    @property
    def pid(self) -> int:
        """Return the worker process id."""
        if self._process.pid is None:
            raise RuntimeError("Worker process has not started.")
        return self._process.pid

    def _call_sync(self, method: str, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self._closed:
                raise RuntimeError("Worker has been closed.")
            request_id = f"worker-{next(self._ids)}"
            self._conn.send(
                {
                    "kind": "call",
                    "id": request_id,
                    "method": method,
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            return self._wait_for_response(request_id)

    async def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Asynchronously send one method call to this worker.

        :param method: Raw worker-side method name.
        :param args: Positional arguments for the raw method.
        :param kwargs: Keyword arguments for the raw method.
        :returns: Worker response payload data.
        """
        return await asyncio.to_thread(self._call_sync, method, *args, **kwargs)

    def _wait_for_startup(self) -> None:
        try:
            deadline = time.monotonic() + 120
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._conn.poll(remaining):
                    raise RuntimeError(
                        "Worker did not report startup within 120 seconds."
                    )
                message = self._conn.recv()
                kind = message.get("kind")
                if kind == "dependency_call":
                    self._serve_dependency_call(message)
                    continue
                if kind == "dependency_batch_call":
                    self._serve_dependency_batch_call(message)
                    continue
                if kind != "started":
                    raise RuntimeError(
                        f"Unexpected worker startup message: {message!r}"
                    )
                if not message.get("ok", False):
                    raise_runtime_error(message)
                return
        except Exception:
            self._closed = True
            try:
                self._conn.close()
            except Exception:
                pass
            self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2)
            raise
        return

    def _wait_for_response(self, request_id: str) -> Any:
        while True:
            message = self._conn.recv()
            kind = message.get("kind")
            if kind == "dependency_call":
                self._serve_dependency_call(message)
                continue
            if kind == "dependency_batch_call":
                self._serve_dependency_batch_call(message)
                continue
            if kind == "response" and message.get("id") == request_id:
                if not message.get("ok", False):
                    raise_runtime_error(message)
                return message.get("data")
            raise RuntimeError(f"Unexpected worker message: {message!r}")

    def _serve_dependency_call(self, message: dict[str, Any]) -> None:
        """Serve one worker dependency call against the parent manager."""
        try:
            handle = self._manager.get(message["resource"])
            method = message["method"]
            if method == "__getattr__":
                result = getattr(handle, message["args"][0])
            else:
                result = getattr(handle, method)(
                    *message.get("args", ()),
                    **message.get("kwargs", {}),
                )
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            response = {
                "kind": "dependency_response",
                "id": message["id"],
                "ok": True,
                "data": result,
            }
        except Exception as exc:  # noqa: BLE001
            response = serialize_error(exc)
            response.update({"kind": "dependency_response", "id": message["id"]})
        self._conn.send(response)
        return

    def _serve_dependency_batch_call(self, message: dict[str, Any]) -> None:
        """Serve one worker dependency batch call against the parent manager."""
        try:
            handle = self._manager.get(message["resource"])
            result = handle._target.batch_call(
                message["calls"],
                log_interval=message.get("log_interval", 0),
                display=message.get("display", "none"),
                desc=message.get("desc", "Calling"),
            )
            response = {
                "kind": "dependency_response",
                "id": message["id"],
                "ok": True,
                "data": result,
            }
        except Exception as exc:  # noqa: BLE001
            response = serialize_error(exc)
            response.update({"kind": "dependency_response", "id": message["id"]})
        self._conn.send(response)
        return

    def close(self) -> None:
        """Close the worker process and terminate it if graceful close fails."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            request_id = f"close-{next(self._ids)}"
            try:
                self._conn.send({"kind": "close", "id": request_id})
                if self._conn.poll(2):
                    self._conn.recv()
            except Exception:
                pass
            finally:
                try:
                    self._conn.close()
                except Exception:
                    pass

        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2)
        return


class ProcessWorkerPool:
    """Pool of replicated worker processes for one process target.

    The primary worker is used for attribute reads. Primitive calls use the next
    available worker when the pool has replicas, preserving one in-flight call
    per worker.
    """

    def __init__(
        self,
        raw_cls: type[Any],
        config: Any,
        refs: dict[str, ResourceRefDescriptor],
        manager: ResourceManager,
        worker_count: int,
        *,
        worker_env_updates: tuple[dict[str, str], ...] | None = None,
    ) -> None:
        """Spawn a worker pool.

        :param raw_cls: Raw resource class to instantiate in each worker.
        :param config: Config object sent to each worker constructor.
        :param refs: Ref descriptors materialized as parent-proxy handles.
        :param manager: Parent manager used to serve dependency calls.
        :param worker_count: Number of worker processes to spawn.
        :param worker_env_updates: Optional per-worker environment updates.
        :raises ValueError: If worker counts are invalid.
        """
        if worker_count < 1:
            raise ValueError("worker_count must be greater than or equal to 1.")
        if worker_env_updates is None:
            env_updates_by_worker = (None,) * worker_count
        else:
            if len(worker_env_updates) != worker_count:
                raise ValueError(
                    "worker_env_updates length must match worker_count."
                )
            env_updates_by_worker = worker_env_updates
        self._workers = [
            ProcessWorkerClient(
                raw_cls,
                config,
                refs,
                manager,
                env_updates=env_updates,
            )
            for env_updates in env_updates_by_worker
        ]
        self._available_workers: asyncio.Queue[ProcessWorkerClient] = asyncio.Queue()
        for worker in self._workers:
            self._available_workers.put_nowait(worker)
        return

    @property
    def primary_pid(self) -> int:
        """Return the primary worker process id."""
        return self._workers[0].pid

    async def call_primary(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call the primary worker."""
        return await self._workers[0].call(method, *args, **kwargs)

    async def call_available(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call the next available worker, falling back to primary for one worker."""
        if len(self._workers) == 1:
            return await self.call_primary(method, *args, **kwargs)

        worker = await self._available_workers.get()
        try:
            return await worker.call(method, *args, **kwargs)
        finally:
            self._available_workers.put_nowait(worker)

    async def async_close(self) -> None:
        """Asynchronously close all workers."""
        for worker in self._workers:
            try:
                await asyncio.to_thread(worker.close)
            except Exception:
                pass
        return
