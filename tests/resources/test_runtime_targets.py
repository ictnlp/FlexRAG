from __future__ import annotations

import os
import time

import pytest

from flexrag.common.dataclasses import Context
from flexrag.resources import ResourceManager, ResourceSpec, RuntimeCallError
from tests.resources.support.registry import FAKE_RESOURCES


def contexts() -> list[Context]:
    return [
        Context(context_id="doc-a", data={"text": "alpha"}),
        Context(context_id="doc-b", data={"text": "beta"}),
        Context(context_id="doc-c", data={"text": "gamma"}),
    ]


def manager(
    *,
    encoder_config: dict | None = None,
    encoder_runtime: str = "direct",
    encoder_options: dict | None = None,
    backend_config: dict | None = None,
) -> ResourceManager:
    return ResourceManager(
        [
            ResourceSpec(
                name="encoder",
                resource_name="fake_encoder",
                runtime=encoder_runtime,
                config=encoder_config or {},
                runtime_options=encoder_options or {},
            ),
            ResourceSpec(
                name="backend",
                resource_name="fake_collection_backend",
                config=backend_config or {},
                refs={"encoder": "encoder"},
            ),
        ],
        registry=FAKE_RESOURCES,
    )


def runtime_pid(handle: object) -> int:
    return handle._target.call("runtime_pid")


def runtime_env(handle: object) -> dict[str, str | None]:
    return handle._target.call("runtime_env")


def test_direct_target_smoke_and_handle_lifecycle_boundary() -> None:
    resources = manager()
    try:
        encoder = resources.get("encoder")

        assert encoder.encode("alpha").tolist() == [[5.0, 13.0]]
        assert runtime_pid(encoder) == os.getpid()
        assert not hasattr(encoder, "close")
        assert not hasattr(encoder, "async_close")
    finally:
        resources.close()


def test_process_target_parent_proxy_and_dependency_errors() -> None:
    resources = manager()
    try:
        backend = resources.get("backend")

        assert runtime_pid(backend) != os.getpid()
        backend.rebuild(contexts())
        assert backend.count() == 3
        assert backend.search_hits(["alpha"], top_k=1)[0][0].context_id == "doc-a"
    finally:
        resources.close()

    resources = manager(encoder_config={"fail_on": "boom"})
    try:
        backend = resources.get("backend")
        with pytest.raises(RuntimeCallError, match="FakeEncoder failed"):
            backend.rebuild([Context(context_id="bad", data={"text": "boom"})])
    finally:
        resources.close()


def test_process_worker_startup_can_call_parent_dependency_ref() -> None:
    resources = manager(backend_config={"startup": True})
    try:
        backend = resources.get("backend")
        state = backend._target.getattr("startup")

        assert state["pid"] != os.getpid()
        assert state["embedding_size"] == 2
        assert state["vector"] == [7.0, 80.0]
    finally:
        resources.close()


@pytest.mark.asyncio
async def test_process_pool_runs_parallel_batches_in_order() -> None:
    resources = manager(
        encoder_config={"delay_seconds": 0.2},
        encoder_runtime="process",
        encoder_options={"batch_size": 1, "worker_count": 2},
    )
    try:
        encoder = resources.get("encoder")
        assert len({runtime_pid(encoder) for _ in range(4)}) == 2

        start = time.monotonic()
        embeddings = await encoder.async_encode(["gamma", "alpha", "beta", "delta"])
        elapsed = time.monotonic() - start

        assert embeddings.tolist() == [
            [5.0, 10.0],
            [5.0, 13.0],
            [4.0, 8.0],
            [5.0, 17.0],
        ]
        assert elapsed < 0.65
    finally:
        await resources.async_close()


@pytest.mark.parametrize(
    ("device_groups", "env_key", "expected"),
    [
        ([["cuda:0"], ["cuda:1"]], "runtime_cuda_visible_devices", {"0", "1"}),
        ([["xpu:0"], ["xpu:1"]], "runtime_ze_affinity_mask", {"0", "1"}),
    ],
)
def test_process_device_groups_set_worker_accelerator_env(
    monkeypatch,
    device_groups: list[list[str]],
    env_key: str,
    expected: set[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "8")
    resources = manager(
        encoder_runtime="process",
        encoder_options={"device_groups": device_groups},
    )
    try:
        encoder = resources.get("encoder")
        envs = [runtime_env(encoder) for _ in range(4)]

        assert {env[env_key] for env in envs} == expected
        import_key = env_key.replace("runtime_", "import_", 1)
        assert {env[import_key] for env in envs} == expected
    finally:
        resources.close()


def test_async_target_prefers_async_twin_and_limits_retry_attempts() -> None:
    resources = ResourceManager(
        [
            ResourceSpec(
                name="encoder",
                resource_name="fake_async_encoder",
                config={"failures_before_success": 1},
                runtime_options={
                    "retry_times": 1,
                    "retry_min_delay": 0,
                    "retry_max_delay": 0,
                    "rpm": 600,
                },
            )
        ],
        registry=FAKE_RESOURCES,
    )
    try:
        encoder = resources.get("encoder")
        start = time.monotonic()

        assert encoder.encode("alpha").tolist() == [[5.0, 13.0]]
        assert time.monotonic() - start >= 0.09
        assert encoder._target.call("call_counts") == {"sync": 0, "async": 2}
    finally:
        resources.close()


@pytest.mark.parametrize(
    ("config", "options", "error"),
    [
        ({"failures_before_success": 2}, {"retry_times": 1}, ValueError),
        ({"delay_seconds": 0.2}, {"timeout": 0.05}, TimeoutError),
    ],
)
def test_async_target_propagates_retry_and_timeout_errors(
    config: dict,
    options: dict,
    error: type[Exception],
) -> None:
    resources = ResourceManager(
        [
            ResourceSpec(
                name="encoder",
                resource_name="fake_async_encoder",
                config=config,
                runtime_options=options,
            )
        ],
        registry=FAKE_RESOURCES,
    )
    try:
        with pytest.raises(error):
            resources.get("encoder").encode("alpha")
    finally:
        resources.close()


@pytest.mark.asyncio
async def test_batch_scheduler_concurrency_progress_and_empty_input(mocker) -> None:
    records: list[dict] = []

    class ProgressRecorder:
        def __init__(self, *, total, interval, display):
            self.record = {"total": total, "interval": interval, "display": display, "updates": []}
            records.append(self.record)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def update(self, step=1, desc=None):
            self.record["updates"].append((step, desc))

    mocker.patch(
        "flexrag.resources.runtime.scheduler.SimpleProgressLogger",
        ProgressRecorder,
    )
    resources = ResourceManager(
        [
            ResourceSpec(
                name="encoder",
                resource_name="fake_encoder",
                config={"delay_seconds": 0.2},
                runtime_options={"batch_size": 1, "max_concurrency": 2},
            )
        ],
        registry=FAKE_RESOURCES,
    )
    try:
        encoder = resources.get("encoder")
        start = time.monotonic()
        embeddings = await encoder.async_encode(
            ["alpha", "beta"],
            log_interval=7,
            display="none",
        )

        assert embeddings.tolist() == [[5.0, 13.0], [4.0, 8.0]]
        assert time.monotonic() - start < 0.35
        assert records == [
            {
                "total": 2,
                "interval": 7,
                "display": "none",
                "updates": [(1, "Encoding"), (1, "Encoding")],
            }
        ]
        records.clear()
        assert encoder.encode([]).shape == (0, 2)
        assert records == []
    finally:
        await resources.async_close()


@pytest.mark.parametrize(
    "spec",
    [
        ResourceSpec(
            name="encoder",
            resource_name="fake_encoder",
            runtime_options={"batch_size": 0},
        ),
        ResourceSpec(
            name="encoder",
            resource_name="fake_encoder",
            runtime_options={"worker_count": 2},
        ),
        ResourceSpec(
            name="encoder",
            resource_name="fake_encoder",
            runtime="process",
            runtime_options={"timeout": 1},
        ),
        ResourceSpec(
            name="encoder",
            resource_name="fake_async_encoder",
            runtime_options={"device_groups": [["cuda:0"]]},
        ),
        ResourceSpec(
            name="backend",
            resource_name="fake_collection_backend",
            runtime_options={"worker_count": 2},
        ),
        ResourceSpec(
            name="encoder",
            resource_name="fake_encoder",
            runtime="process",
            runtime_options={"device_groups": [["cuda:0"], ["xpu:0"]]},
        ),
    ],
)
def test_runtime_options_reject_representative_invalid_specs(
    spec: ResourceSpec,
) -> None:
    resources = ResourceManager([spec], registry=FAKE_RESOURCES)
    with pytest.raises(ValueError):
        resources.get(spec.name)
