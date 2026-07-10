from __future__ import annotations

from flexrag.resources.registry import _ResourceRegister

from .fakes import (
    FakeAsyncEncoder,
    FakeCollectionBackend,
    FakeEncoder,
    FakeEncoderConfig,
    FakeGenerator,
)

FakeScorerConfig = type("FakeScorerConfig", (), {})
FAKE_RESOURCES = _ResourceRegister()

FAKE_RESOURCES.register(
    "fake_encoder",
    interface="encoder",
    config_class=FakeEncoderConfig,
    parallel_safe=True,
)(FakeEncoder)
FAKE_RESOURCES.register(
    "fake_async_encoder",
    interface="encoder",
    default_runtime="async",
    parallel_safe=True,
)(FakeAsyncEncoder)
FAKE_RESOURCES.register(
    "fake_generator",
    interface="generator",
    parallel_safe=True,
    batching=False,
)(FakeGenerator)
FAKE_RESOURCES.register(
    "fake_scorer",
    interface="scorer",
    config_class=FakeScorerConfig,
    parallel_safe=True,
)(FakeGenerator)
FAKE_RESOURCES.register(
    "fake_collection_backend",
    interface="collection_backend",
    default_runtime="process",
)(FakeCollectionBackend)
