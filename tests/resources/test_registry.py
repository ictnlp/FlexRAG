from __future__ import annotations

import pytest

from flexrag.resources import _ResourceRegister


def test_resource_register_resolves_by_name_and_config() -> None:
    class Config:
        pass

    registry = _ResourceRegister()

    @registry.register(
        "custom_encoder",
        interface="encoder",
        config_class=Config,
        default_runtime="process",
        parallel_safe=True,
        batching=False,
    )
    class CustomEncoder:
        pass

    entry = registry.resolve_name("custom_encoder")

    assert entry.resource_name == "custom_encoder"
    assert entry.raw_cls is CustomEncoder
    assert entry.interface == "encoder"
    assert entry.default_runtime == "process"
    assert entry.parallel_safe is True
    assert entry.batching is False
    assert registry.resolve_config(Config()) is entry
    assert registry.names == ("custom_encoder",)
    assert registry.entries == (entry,)


def test_resource_register_rejects_conflicting_entries() -> None:
    class Config:
        pass

    registry = _ResourceRegister()
    registry.register("custom", interface="encoder", config_class=Config)(
        type("CustomEncoder", (), {})
    )

    with pytest.raises(ValueError, match="already registered"):
        registry.register("custom", interface="encoder")(type("Other", (), {}))
    with pytest.raises(ValueError, match="already registered"):
        registry.register("other", interface="encoder", config_class=Config)(
            type("Other", (), {})
        )


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"resource_name": "", "interface": "encoder"}, ValueError),
        ({"resource_name": "x", "interface": ""}, ValueError),
        ({"resource_name": "x", "interface": "encoder", "default_runtime": "remote"}, ValueError),
        ({"resource_name": "x", "interface": "encoder", "config_class": object()}, TypeError),
    ],
)
def test_resource_register_rejects_invalid_entries(kwargs: dict, error: type[Exception]):
    registry = _ResourceRegister()
    resource_name = kwargs.pop("resource_name")
    with pytest.raises(error):
        registry.register(resource_name, **kwargs)
