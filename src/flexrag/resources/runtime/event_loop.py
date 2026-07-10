from __future__ import annotations

import asyncio
import atexit
import threading
from concurrent.futures import Future, TimeoutError
from typing import Any


class BackgroundEventLoop:
    """Background asyncio loop used by runtime targets.

    Runtime targets submit both synchronous and asynchronous public calls to
    this loop so they share the same scheduler and call policy implementation.
    """

    def __init__(self) -> None:
        """Create a lazily-started background event loop."""
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = threading.Event()
        self._started = False
        self._stopped = False
        self._lock = threading.Lock()
        atexit.register(self.stop)
        return

    def run_async(self, coro: Any) -> Future:
        """Submit a coroutine to the background loop.

        :param coro: Coroutine object to run.
        :returns: Thread-safe future for the submitted coroutine.
        :raises RuntimeError: If the loop has been stopped.
        """
        self._ensure_started()
        if self._loop is None:
            raise RuntimeError("Background event loop has not started.")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _ensure_started(self) -> None:
        with self._lock:
            if self._stopped:
                raise RuntimeError("Background event loop has been stopped.")
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self._ready.wait()
        return

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()
        return

    async def _shutdown(self) -> None:
        loop = asyncio.get_running_loop()
        current = asyncio.current_task(loop=loop)
        tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if task is not current and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await loop.shutdown_asyncgens()
        if hasattr(loop, "shutdown_default_executor"):
            await loop.shutdown_default_executor()
        return

    def stop(self) -> None:
        """Stop the background loop and cancel pending tasks."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        try:
            atexit.unregister(self.stop)
        except Exception:
            pass

        loop = self._loop
        thread = self._thread
        if loop is None or thread is None:
            return

        if loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
                future.result(timeout=5)
            except TimeoutError:
                pass
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except RuntimeError:
                pass

        if thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5)
        return
