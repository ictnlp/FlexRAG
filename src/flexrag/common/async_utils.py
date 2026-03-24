import asyncio
import atexit
import threading
from concurrent.futures import Future, TimeoutError


class BackgroundEventLoop:
    """A background thread running an asyncio loop, created lazily."""

    def __init__(self):
        self._thread = None
        self._loop = None
        self._ready = threading.Event()
        self._started = False
        self._stopped = False
        atexit.register(self.stop)

    def ensure_started(self):
        """Lazy start the background event loop."""
        if self._stopped:
            raise RuntimeError("BackgroundEventLoop has been stopped.")
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

    async def _shutdown(self):
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

    def run_async(self, coro) -> Future:
        """Submit coroutine safely to the background loop."""
        self.ensure_started()
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self):
        if self._stopped:
            return
        self._stopped = True

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
