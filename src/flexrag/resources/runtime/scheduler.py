from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from flexrag.common import SimpleProgressLogger

from .base import RuntimeCall

ExecuteCall = Callable[[RuntimeCall], Awaitable[Any]]
AttemptCall = Callable[[], Awaitable[Any]]


class RateLimiter:
    """Attempt-level request-per-minute limiter.

    The limiter is shared by a target call policy. It delays each primitive
    attempt before execution and does not know about handle-level batches.
    """

    def __init__(self, *, rpm: float = 0) -> None:
        """Create a limiter.

        :param rpm: Maximum attempts per minute. ``0`` disables limiting.
        :raises ValueError: If ``rpm`` is negative.
        """
        if rpm < 0:
            raise ValueError("rpm must be non-negative.")
        self.rpm = float(rpm)
        self._lock: asyncio.Lock | None = None
        self._next_request_time = 0.0
        return

    async def wait(self) -> None:
        """Wait until the next attempt may start."""
        if self.rpm <= 0:
            return
        if self._lock is None:
            self._lock = asyncio.Lock()
        interval = 60.0 / self.rpm
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            if self._next_request_time > now:
                await asyncio.sleep(self._next_request_time - now)
                now = loop.time()
            self._next_request_time = max(self._next_request_time, now) + interval
        return


class CallPolicy(Protocol):
    """Attempt-level policy applied by the batch scheduler."""

    async def run(self, attempt: AttemptCall) -> Any:
        """Run one primitive attempt under this policy.

        :param attempt: Awaitable factory for the primitive call attempt.
        :returns: Attempt result.
        """
        ...


class NoRetryPolicy:
    """Call policy that only applies rate limiting and never retries."""

    def __init__(self, rate_limiter: RateLimiter | None = None) -> None:
        """Create a no-retry policy.

        :param rate_limiter: Limiter to apply before each attempt. A disabled
            limiter is created when omitted.
        """
        self._rate_limiter = rate_limiter or RateLimiter()
        return

    async def run(self, attempt: AttemptCall) -> Any:
        """Run one attempt after waiting for the rate limiter."""
        await self._rate_limiter.wait()
        return await attempt()


class RetryTimeoutPolicy:
    """Call policy with rate limiting, retry backoff, and per-attempt timeout."""

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        *,
        retry_times: int = 0,
        retry_min_delay: float = 1.0,
        retry_max_delay: float = 60.0,
        timeout: float = 0,
    ) -> None:
        """Create a retry/timeout call policy.

        :param rate_limiter: Limiter to apply before every attempt, including
            retries.
        :param retry_times: Number of retries after the initial attempt.
        :param retry_min_delay: Initial retry backoff delay in seconds.
        :param retry_max_delay: Maximum retry backoff delay in seconds.
        :param timeout: Per-attempt timeout in seconds. ``0`` disables timeout.
        :raises ValueError: If retry or timeout options are invalid.
        """
        if not isinstance(retry_times, int) or isinstance(retry_times, bool):
            raise ValueError("retry_times must be a non-negative integer.")
        if retry_times < 0:
            raise ValueError("retry_times must be a non-negative integer.")
        if retry_min_delay < 0:
            raise ValueError("retry_min_delay must be non-negative.")
        if retry_max_delay < retry_min_delay:
            raise ValueError(
                "retry_max_delay must be greater than or equal to retry_min_delay."
            )
        if timeout < 0:
            raise ValueError("timeout must be non-negative.")
        self._rate_limiter = rate_limiter or RateLimiter()
        self._retry_times = retry_times
        self._retry_min_delay = float(retry_min_delay)
        self._retry_max_delay = float(retry_max_delay)
        self._timeout = float(timeout)
        return

    async def run(self, attempt: AttemptCall) -> Any:
        """Run one primitive call attempt with retries when failures occur."""
        for attempt_idx in range(self._retry_times + 1):
            await self._rate_limiter.wait()
            try:
                coro = attempt()
                if self._timeout > 0:
                    return await asyncio.wait_for(coro, timeout=self._timeout)
                return await coro
            except Exception:
                if attempt_idx >= self._retry_times:
                    raise
                delay = self._retry_delay(attempt_idx)
            if delay > 0:
                await asyncio.sleep(delay)
        raise RuntimeError("Retry policy loop exited unexpectedly.")

    def _retry_delay(self, retry_idx: int) -> float:
        delay = self._retry_min_delay * (2**retry_idx)
        return min(delay, self._retry_max_delay)


class RuntimeBatchScheduler:
    """Batch-level scheduler for primitive runtime calls.

    The scheduler owns concurrency, result ordering, and progress accounting.
    It delegates retry, timeout, and RPM behavior to the supplied call policy.
    """

    def __init__(self, *, max_concurrency: int = 1) -> None:
        """Create a scheduler.

        :param max_concurrency: Maximum primitive calls to run concurrently.
        :raises ValueError: If ``max_concurrency`` is not positive.
        """
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be greater than 0.")
        self.max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None
        return

    async def run(
        self,
        calls: list[RuntimeCall],
        execute_call: ExecuteCall,
        call_policy: CallPolicy,
        *,
        log_interval: int = 0,
        display: str = "none",
        desc: str = "Calling",
    ) -> list[Any]:
        """Execute primitive calls with bounded concurrency.

        :param calls: Calls to execute.
        :param execute_call: Coroutine function that executes one primitive
            call.
        :param call_policy: Attempt-level policy applied to each call.
        :param log_interval: Progress logger update interval.
        :param display: Progress display mode.
        :param desc: Progress description used for completed call weights.
        :returns: Results in the same order as ``calls``.
        """
        if not calls:
            return []
        results: list[Any] = [None] * len(calls)
        total = sum(call.weight for call in calls)
        with SimpleProgressLogger(
            total=total,
            interval=log_interval,
            display=display,
        ) as progress:

            async def _run_one(idx: int, call: RuntimeCall) -> None:
                semaphore = self._get_semaphore()
                async with semaphore:
                    results[idx] = await call_policy.run(lambda: execute_call(call))
                progress.update(call.weight, desc=desc)
                return

            await asyncio.gather(
                *(_run_one(idx, call) for idx, call in enumerate(calls))
            )
        return results

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        return self._semaphore
