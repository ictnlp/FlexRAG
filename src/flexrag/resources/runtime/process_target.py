from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..refs import ResourceRefDescriptor
from .base import RuntimeCall
from .process_runtime import ProcessWorkerPool
from .scheduler import NoRetryPolicy, RateLimiter
from .target_base import RuntimeTargetBase

if TYPE_CHECKING:
    from ..resource_manager import ResourceManager


class ProcessTarget(RuntimeTargetBase):
    """Runtime target backed by one or more worker processes.

    ``ProcessTarget`` constructs raw resources inside spawned workers and
    passes refs as serializable descriptors. Workers materialize those refs as
    parent-proxy handles, so worker-side resources can call parent-managed
    dependencies without owning their lifecycle. A single worker represents a
    stateful owner; multiple workers represent replicated parallel-safe
    resources.
    """

    def __init__(
        self,
        raw_cls: type[Any],
        config: Any,
        refs: dict[str, ResourceRefDescriptor],
        manager: ResourceManager,
        *,
        worker_count: int = 1,
        worker_env_updates: tuple[dict[str, str], ...] | None = None,
        batch_size: int = 1,
        max_concurrency: int | None = None,
        rpm: float = 0,
    ) -> None:
        """Create a process-backed target.

        :param raw_cls: Raw resource class to instantiate in each worker.
        :param config: Config object sent to each worker constructor.
        :param refs: Serializable ref descriptors for worker-side proxy handles.
        :param manager: Parent resource manager that serves dependency calls.
        :param worker_count: Number of worker processes to spawn.
        :param worker_env_updates: Optional per-worker environment updates, used
            for accelerator visibility.
        :param batch_size: Public-call batch size exposed to handles.
        :param max_concurrency: Maximum primitive calls to run concurrently. If
            omitted, it defaults to ``worker_count``.
        :param rpm: Attempt-level request-per-minute limit. ``0`` disables
            rate limiting.
        """
        self._pool = ProcessWorkerPool(
            raw_cls,
            config,
            refs,
            manager,
            worker_count,
            worker_env_updates=worker_env_updates,
        )
        super().__init__(
            batch_size=batch_size,
            max_concurrency=max_concurrency or worker_count,
            call_policy=NoRetryPolicy(RateLimiter(rpm=rpm)),
        )
        return

    @property
    def worker_pid(self) -> int:
        """Return the primary worker process id."""
        return self._pool.primary_pid

    async def _async_execute_call(self, call: RuntimeCall) -> Any:
        """Execute a primitive call on the next available worker."""
        return await self._pool.call_available(
            call.method,
            *call.args,
            **call.kwargs,
        )

    async def _async_getattr_impl(self, name: str) -> Any:
        """Read attributes from the primary worker to avoid pool state drift."""
        return await self._pool.call_primary("__getattr__", name)

    async def _async_close_impl(self) -> None:
        """Close the worker pool, swallowing shutdown errors."""
        try:
            await self._pool.async_close()
        except Exception:
            pass
        return
