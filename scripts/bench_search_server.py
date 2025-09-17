#!/usr/bin/env python3
import argparse
import asyncio
import random
import time
from statistics import mean

import httpx
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table


def percentile(sorted_data: list[float], p: float) -> float:
    if not sorted_data:
        return float("nan")
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    if f == c:
        return sorted_data[f]
    d0 = sorted_data[f] * (c - k)
    d1 = sorted_data[c] * (k - f)
    return d0 + d1


async def worker(
    name: int,
    client: httpx.AsyncClient,
    url: str,
    semaphore: asyncio.Semaphore,
    tasks: int,
    queries_pool: list[str],
    queries_per_req: int,
    top_k: int,
    timeout: float,
    latencies_ms: list[float],
    failures: list[str],
    progress: Progress,
    task_id: TaskID,
):
    for _ in range(tasks):
        async with semaphore:
            # compose request
            queries = random.choices(queries_pool, k=queries_per_req)
            payload = {"queries": queries, "top_k": top_k}
            start = time.perf_counter()
            try:
                resp = await client.post(url, json=payload, timeout=timeout)
                if resp.status_code != 200:
                    failures.append(f"HTTP {resp.status_code}")
                else:
                    _ = resp.json()
            except Exception as e:
                failures.append(str(e))
            else:
                lat = (time.perf_counter() - start) * 1000.0
                latencies_ms.append(lat)
            progress.update(task_id, advance=1)
    return


async def main_async(args):
    total_reqs = args.requests
    concurrency = args.concurrency
    queries_per_req = args.queries_per_req

    # prepare queries pool
    if args.queries_file:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            pool = [ln.strip() for ln in f if ln.strip()]
        if not pool:
            pool = [f"query-{i}" for i in range(max(1000, queries_per_req))]
    else:
        pool = [f"query-{i}" for i in range(max(10000, queries_per_req))]

    # configure httpx client
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    transport = httpx.AsyncHTTPTransport(retries=2)

    latencies_ms: list[float] = []
    failures: list[str] = []
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits, transport=transport) as client:
        # warmup
        warmup = max(0, args.warmup)
        if warmup > 0:
            warm_payload = {
                "queries": random.choices(pool, k=queries_per_req),
                "top_k": args.top_k,
            }
            try:
                await client.post(args.url, json=warm_payload, timeout=args.timeout)
            except Exception:
                pass

        # start benchmarking
        with Progress() as progress:
            task_id = progress.add_task("[green]Benchmarking...", total=total_reqs)
            start = time.perf_counter()
            # evenly distribute tasks to each worker
            per_worker = total_reqs // concurrency
            remainder = total_reqs % concurrency
            tasks = []
            for i in range(concurrency):
                count = per_worker + (1 if i < remainder else 0)
                if count == 0:
                    continue
                tasks.append(
                    asyncio.create_task(
                        worker(
                            i,
                            client,
                            args.url,
                            semaphore,
                            count,
                            pool,
                            queries_per_req,
                            args.top_k,
                            args.timeout,
                            latencies_ms,
                            failures=failures,
                            progress=progress,
                            task_id=task_id,
                        )
                    )
                )

            await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - start

    # Summarize results
    succ = len(latencies_ms)
    fail = len(failures)
    total = succ + fail
    latencies_ms.sort()
    rps = succ / elapsed if elapsed > 0 else 0.0

    console = Console()
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="bold magenta")
    data = [
        ("URL", args.url),
        ("Total Requests", f"{total} (Success {succ}, Failure {fail})"),
        ("Concurrency", str(concurrency)),
        ("Queries per Request", str(queries_per_req)),
        ("Top K", str(args.top_k)),
        ("Elapsed Time", f"{elapsed:.3f}s"),
        ("Throughput (RPS)", f"{rps:.2f} req/s"),
    ]
    if succ > 0:
        data.extend(
            [
                ("Average Latency", f"{mean(latencies_ms):.2f} ms"),
                ("p50 Latency", f"{percentile(latencies_ms, 50):.2f} ms"),
                ("p90 Latency", f"{percentile(latencies_ms, 90):.2f} ms"),
                ("p95 Latency", f"{percentile(latencies_ms, 95):.2f} ms"),
                ("p99 Latency", f"{percentile(latencies_ms, 99):.2f} ms"),
            ]
        )
    for k, v in data:
        table.add_row(k, v)
    console.print(table)
    return


def parse_args():
    parser = argparse.ArgumentParser(description="FlexRAG /search server benchmark")
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:3402/search",
        help="The URL of the searcher service.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="The number of concurrent requests.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5000,
        help="The total requests.",
    )
    parser.add_argument(
        "--queries-per-req",
        type=int,
        default=1,
        help="The number of queries in each request.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="top_k",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Single request timeout (seconds).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup requests (not counted in statistics).",
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        default=None,
        help="File containing queries, one per line (optional).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
