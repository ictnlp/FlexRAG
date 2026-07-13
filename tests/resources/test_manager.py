from __future__ import annotations

import pytest

from flexrag.resources import (
    ResourceManager,
    ResourcesConfig,
    ResourceSpec,
    RuntimeConfig,
)
from tests.resources.support.fakes import FakeEncoderConfig
from tests.resources.support.registry import FAKE_RESOURCES


def test_resource_manager_resolves_resource_from_concrete_config() -> None:
    resources = ResourceManager(
        [ResourceSpec(name="encoder", resource_config=FakeEncoderConfig())],
        registry=FAKE_RESOURCES,
    )
    try:
        assert resources.get("encoder").encode("alpha").tolist() == [[5.0, 13.0]]
    finally:
        resources.close()


@pytest.mark.parametrize(
    "spec",
    [
        ResourceSpec(name="encoder"),
        ResourceSpec(
            name="encoder",
            resource_name="fake_scorer",
            resource_config=FakeEncoderConfig(),
        ),
    ],
)
def test_resource_manager_rejects_ambiguous_or_mismatched_specs(
    spec: ResourceSpec,
) -> None:
    resources = ResourceManager([spec], registry=FAKE_RESOURCES)
    with pytest.raises(ValueError):
        resources.get("encoder")


def test_resource_manager_load_preloads_refs_before_roots() -> None:
    resources = ResourceManager.load(
        ResourcesConfig(
            resources=[
                ResourceSpec(name="encoder", resource_name="fake_encoder"),
                ResourceSpec(
                    name="backend",
                    resource_name="fake_collection_backend",
                    refs={"encoders": {"primary": "encoder"}},
                ),
            ],
            preload=["backend"],
        ),
        registry=FAKE_RESOURCES,
    )
    close_order: list[str] = []
    for name, target in resources._targets.items():
        close = target.close

        def record_close(name=name, close=close):
            close_order.append(name)
            close()

        target.close = record_close
    try:
        assert list(resources._handles) == ["encoder", "backend"]
    finally:
        resources.close()
    assert close_order == ["backend", "encoder"]


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (ResourcesConfig(preload=["missing"]), KeyError),
        (
            ResourcesConfig(
                resources=[
                    ResourceSpec(
                        name="first",
                        resource_name="fake_generator",
                        refs={"other": "second"},
                    ),
                    ResourceSpec(
                        name="second",
                        resource_name="fake_generator",
                        refs={"others": {"first": "first"}},
                    ),
                ],
                preload=["first"],
            ),
            ValueError,
        ),
    ],
)
def test_resource_manager_load_rejects_invalid_preload_graph(
    config: ResourcesConfig,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ResourceManager.load(config, registry=FAKE_RESOURCES)


def test_resource_manager_load_closes_loaded_targets_after_preload_failure() -> None:
    observed_target = None
    original_close = ResourceManager.close

    def record_close(self):
        nonlocal observed_target
        observed_target = self._targets["encoder"]
        return original_close(self)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(ResourceManager, "close", record_close)
        with pytest.raises(ValueError, match="batched public calls"):
            ResourceManager.load(
                ResourcesConfig(
                    resources=[
                        ResourceSpec(name="encoder", resource_name="fake_encoder"),
                        ResourceSpec(
                            name="generator",
                            resource_name="fake_generator",
                            runtime_config=RuntimeConfig(batch_size=2),
                        ),
                    ],
                    preload=["encoder", "generator"],
                ),
                registry=FAKE_RESOURCES,
            )

    assert observed_target is not None
    with pytest.raises(RuntimeError, match="DirectTarget has been closed"):
        observed_target.call("runtime_pid")


def test_resource_manager_close_clears_cached_state_and_is_idempotent() -> None:
    resources = ResourceManager(
        [ResourceSpec(name="encoder", resource_name="fake_encoder")],
        registry=FAKE_RESOURCES,
    )
    resources.get("encoder")

    resources.close()
    resources.close()

    assert resources._handles == {}
    assert resources._targets == {}
    assert resources._load_order == []
    with pytest.raises(RuntimeError, match="ResourceManager has been closed"):
        resources.get("encoder")


@pytest.mark.asyncio
async def test_resource_manager_async_context_manager_closes() -> None:
    resources = ResourceManager(
        [ResourceSpec(name="encoder", resource_name="fake_encoder")],
        registry=FAKE_RESOURCES,
    )

    async with resources as manager:
        manager.get("encoder")

    assert resources._handles == {}
    with pytest.raises(RuntimeError, match="ResourceManager has been closed"):
        resources.get("encoder")


def test_resources_config_round_trip_uses_new_schema() -> None:
    config = ResourcesConfig.loads(
        """
resources:
  - name: encoder
    resource_name: fake_encoder
    resource_config:
      delay_seconds: 0.0
    runtime_config:
      name: direct
      batch_size: 8
  - name: composite
    resource_name: fake_generator
    refs:
      encoders:
        primary: encoder
preload:
  - encoder
"""
    )

    loaded = ResourcesConfig.loads(config.dumps())

    assert loaded == config
    assert loaded.resources[0].resource_name == "fake_encoder"
    assert loaded.resources[0].runtime_config == RuntimeConfig(
        name="direct",
        batch_size=8,
    )
    assert loaded.resources[1].refs == {"encoders": {"primary": "encoder"}}
