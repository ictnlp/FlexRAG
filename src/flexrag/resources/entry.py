from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ResourceEntry:
    """Registry metadata for a resource type.

    Entries connect declarative ``ResourceSpec`` values to raw classes and typed
    handles. They intentionally describe resource shape and runtime defaults,
    not per-call execution behavior.

    :param resource_name: Canonical registry name used by ``ResourceSpec``.
    :param raw_cls: Raw resource class constructed by the selected target.
    :param interface: Interface name used to choose the typed handle class.
    :param config_class: Optional config dataclass used to materialize dict
        specs before construction.
    :param default_runtime: Runtime name used when a spec omits ``runtime``.
    :param parallel_safe: Whether process runtime may replicate this resource
        across multiple workers.
    :param batching: Whether public handle calls may pass multiple samples to
        one raw resource call.
    """

    resource_name: str
    raw_cls: type[Any]
    interface: str
    config_class: type[Any] | None = None
    default_runtime: str = "direct"
    parallel_safe: bool = False
    batching: bool = True
