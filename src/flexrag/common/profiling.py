import asyncio
import gc
import inspect
import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Generator, Optional

import numpy as np
from rich.console import Console
from rich.table import Table

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


class Span:
    """Pure data-bearing node, includes timestamps and parent-child relationships."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = time.perf_counter()
        self.end_time = 0.0
        self.children: list["Span"] = []

        try:
            task = asyncio.current_task()
            self.tid = id(task) if task else threading.get_ident()
        except RuntimeError:
            self.tid = threading.get_ident()

        self.parent = _current_span.get()
        if self.parent is not None:
            self.parent.children.append(self)

    def end(self) -> None:
        self.end_time = time.perf_counter()

    @property
    def duration(self) -> float:
        if self.end_time != 0.0:
            return self.end_time - self.start_time
        return time.perf_counter() - self.start_time


class ProfilerSession:
    """Session Container for collecting spans and computing aggregated stats."""

    def __init__(self):
        self.stats = defaultdict(list)
        self._spans: list[Span] = []
        return

    def record_span(self, section_name: str, span: Span) -> None:
        self.stats[section_name].append(span.duration)
        self._spans.append(span)
        return

    def summary(self, show: bool = True) -> dict[str, dict[str, int | float]]:
        summary = {}
        for name, durations in self.stats.items():
            wall_clock = np.asarray(durations, dtype=float)
            summary[name] = {
                "Calls": len(durations),
                "Wall Avg": float(wall_clock.mean()),
                "Wall Median": float(np.median(wall_clock)),
                "Wall P95": float(np.percentile(wall_clock, 95)),
                "Wall Sum": float(wall_clock.sum()),
            }
        if show:
            console = Console()
            table = Table(title="Profiler Summary")
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Calls", justify="right")
            table.add_column("Wall Avg", justify="right")
            table.add_column("Wall Median", justify="right")
            table.add_column("Wall P95", justify="right")
            table.add_column("Wall Sum", justify="right")
            for name, st in summary.items():
                table.add_row(
                    name,
                    str(st["Calls"]),
                    f"{st['Wall Avg']:.4f}",
                    f"{st['Wall Median']:.4f}",
                    f"{st['Wall P95']:.4f}",
                    f"{st['Wall Sum']:.4f}",
                )
            console.print(table)
        return summary

    def export_chrome_trace(self, filepath: str) -> None:
        events = []
        pid = os.getpid()
        for span in sorted(self._spans, key=lambda item: item.start_time):
            events.append(
                {
                    "name": span.name,
                    "cat": "PERF",
                    "ph": "X",
                    "pid": pid,
                    "tid": span.tid,
                    "ts": span.start_time * 1e6,
                    "dur": span.duration * 1e6,
                }
            )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(events, f)
        print(f"\n[Profiler] Flame graph trace exported to {filepath}.")
        return


_active_session: ContextVar[Optional[ProfilerSession]] = ContextVar(
    "active_session", default=None
)
_current_span: ContextVar[Optional[Span]] = ContextVar("current_span", default=None)


def _maybe_cuda_sync(device_id: int | None = None) -> None:
    if has_torch and torch.cuda.is_available():
        torch.cuda.synchronize(device_id)


@contextmanager
def start_session(
    *, disable_gc: bool = False
) -> Generator[ProfilerSession, None, None]:
    """Start a profiling session."""
    session = ProfilerSession()
    token = _active_session.set(session)

    gc_disabled_here = disable_gc and gc.isenabled()
    if gc_disabled_here:
        gc.disable()

    try:
        yield session
    finally:
        if gc_disabled_here:
            gc.enable()
        _active_session.reset(token)


@contextmanager
def record_span(
    name: str,
    *,
    sync_cuda: bool = False,
    device_id: int | None = None,
) -> Generator[Optional[Span], None, None]:
    """Record a span under the current session. If no session is active, this is a no-op."""
    session = _active_session.get()

    if session is None:
        yield None
        return

    if sync_cuda:
        _maybe_cuda_sync(device_id)

    span = Span(name)
    token = _current_span.set(span)
    try:
        yield span
    finally:
        if sync_cuda:
            _maybe_cuda_sync(device_id)
        span.end()
        _current_span.reset(token)
        session.record_span(name, span)


def trace(
    name: str,
    *,
    sync_cuda: bool = False,
    device_id: int | None = None,
) -> Callable:
    """Decorator to trace a function."""

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with record_span(
                    name,
                    sync_cuda=sync_cuda,
                    device_id=device_id,
                ):
                    return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with record_span(
                name,
                sync_cuda=sync_cuda,
                device_id=device_id,
            ):
                return func(*args, **kwargs)

        return sync_wrapper

    return decorator
