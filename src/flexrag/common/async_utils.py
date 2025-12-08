import asyncio
import threading
from concurrent.futures import Future


class BackgroundEventLoop:
    """A background thread running an asyncio loop, created lazily."""

    def __init__(self):
        self._thread = None
        self._loop = None
        self._ready = threading.Event()
        self._started = False

    def ensure_started(self):
        """Lazy start the background event loop."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()

        try:
            loop.run_forever()
        finally:
            loop.close()

    def run_async(self, coro) -> Future:
        """Submit coroutine safely to the background loop."""
        self.ensure_started()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join()
