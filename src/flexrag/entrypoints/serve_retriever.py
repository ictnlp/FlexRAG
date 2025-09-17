import asyncio
from dataclasses import asdict
from typing import Any, Optional

import hydra
import uvicorn
from fastapi import FastAPI, HTTPException
from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from flexrag.retriever import FlexRetriever, FlexRetrieverConfig
from flexrag.utils import LOGGER_MANAGER, configure, extract_config

app = FastAPI()


retriever: FlexRetriever


class SearchRequest(BaseModel):
    """Request model for /search endpoint."""

    queries: list[str] = Field(
        description="List of queries to search for. Each query should be a string.",
    )
    top_k: int = Field(
        default=10,
        description="Number of top results to return.",
    )


class _RequestItem:
    """An individual request item containing query, top_k, and a future for the result."""

    def __init__(self, query: str, top_k: int):
        self.query = query
        self.top_k = top_k
        self.future: asyncio.Future = asyncio.get_running_loop().create_future()
        return


class _MicroBatchBuffer:
    """A buffer for accumulating queries and their top_k values."""

    def __init__(self) -> None:
        self.requests: list[_RequestItem] = []
        return

    def add(self, requests: list[_RequestItem]) -> None:
        self.requests.extend(requests)
        return

    def clear(self) -> None:
        self.requests = []
        return

    def is_empty(self) -> bool:
        return len(self.requests) == 0

    @property
    def max_top_k(self) -> int:
        return max(item.top_k for item in self.requests) if self.requests else 1

    def __len__(self) -> int:
        return len(self.requests)


class MicroBatcher:
    """
    Aggregate multiple incoming requests into a single retriever call
    to improve GPU/CPU utilization and throughput.

    Strategy:
    - Collect requests up to `max_batch_size` or `max_latency_ms`.
    - Merge queries; call retriever.search once with top_k = max(top_k_i).
    - Split and trim results per-request; resolve futures.
    """

    def __init__(
        self,
        max_batch_size: int = 128,
        max_latency_ms: int = 5,
    ) -> None:
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_latency = max(1, int(max_latency_ms)) / 1000.0
        self._lock = asyncio.Lock()
        self._batch_buffer: _MicroBatchBuffer = _MicroBatchBuffer()
        self._timer: Optional[asyncio.Task] = None
        return

    async def submit(
        self, queries: list[str], top_k: int
    ) -> list[list[dict[str, Any]]]:
        requests = [_RequestItem(q, top_k) for q in queries]
        async with self._lock:
            remaining = len(requests)
            while remaining > 0:
                space = self.max_batch_size - len(self._batch_buffer)
                take = min(space, remaining)
                p = -remaining + take if remaining - take > 0 else None
                self._batch_buffer.add(requests[-remaining:p])
                remaining -= take
                if len(self._batch_buffer) >= self.max_batch_size:
                    await self._flush_locked()
            # Flush after max_latency_ms if not already scheduled
            if len(self._batch_buffer) > 0 and self._timer is None:
                self._timer = asyncio.create_task(self._flush_after_delay())
        # Wait for result
        return await asyncio.gather(
            *(item.future for item in requests),
            return_exceptions=False,
        )

    async def _flush_after_delay(self) -> None:
        try:
            await asyncio.sleep(self.max_latency)
            async with self._lock:
                await self._flush_locked()
        except asyncio.CancelledError:
            pass

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    async def _flush_locked(self) -> None:
        if self._batch_buffer.is_empty():
            self._cancel_timer()
            return
        self._timer = None

        # Snapshot and clear buffer
        batch = self._batch_buffer.requests
        max_top_k = self._batch_buffer.max_top_k
        queries = [item.query for item in batch]
        self._batch_buffer = _MicroBatchBuffer()

        try:
            results = await retriever.async_search(query=queries, top_k=max_top_k)
            # Split back per request and trim to each requested top_k
            for idx, req in enumerate(batch):
                trimmed = results[idx][: req.top_k]
                # Convert to plain dicts for JSON response
                payload = [asdict(r) for r in trimmed]
                if not req.future.done():
                    req.future.set_result(payload)
        except Exception as e:  # noqa: BLE001
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(e)
        return


micro_batcher: Optional[MicroBatcher] = None


@configure
class Config(FlexRetrieverConfig):
    host: str = "0.0.0.0"
    port: int = 3402
    # Throughput tuning
    max_batch_size: int = Field(
        default=128,
        description="Maximum number of requests to batch together.",
    )
    max_latency_ms: int = Field(
        default=5,
        description="Maximum latency in milliseconds to wait for batching requests.",
    )
    workers: int = Field(
        default=1,
        description="Number of Uvicorn worker processes.",
    )


cs = ConfigStore.instance()
cs.store(name="default", node=Config)
logger = LOGGER_MANAGER.get_logger("serve_retriever")


@app.post("/search")
async def search(args: SearchRequest):
    try:
        return await micro_batcher.submit(queries=args.queries, top_k=args.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config):
    cfg = extract_config(cfg, Config)
    global retriever
    global micro_batcher

    # Init retriever
    retriever = FlexRetriever(cfg)

    # Init micro-batcher if enabled
    micro_batcher = MicroBatcher(
        max_batch_size=cfg.max_batch_size,
        max_latency_ms=cfg.max_latency_ms,
    )

    # Prefer uvloop/httptools if available for better I/O perf
    uvicorn_kwargs: dict[str, Any] = {}
    try:
        import uvloop  # type: ignore

        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        uvicorn_kwargs["loop"] = "uvloop"
    except Exception:
        pass
    try:
        import httptools  # type: ignore  # noqa: F401

        uvicorn_kwargs["http"] = "httptools"
    except Exception:
        pass

    uvicorn.run(
        app,
        host=cfg.host,
        port=cfg.port,
        workers=max(1, int(cfg.workers)),
        **uvicorn_kwargs,
    )


if __name__ == "__main__":
    main()
