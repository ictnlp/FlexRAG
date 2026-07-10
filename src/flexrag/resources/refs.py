from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flexrag.common import configure


@dataclass(frozen=True)
class ResourceRefDescriptor:
    """Serializable description of a resource ref for process workers.

    Process workers cannot receive parent-process handles directly. The parent
    sends this descriptor instead, and the worker rebuilds a typed handle around
    a ``ParentProxyTarget``.

    :param name: Resource name in the parent manager.
    :param interface: Resource interface used to choose the typed handle class.
    :param batch_size: Target batch size visible through the proxy handle.
    :param batching: Whether the referenced handle supports batched public
        calls.
    """

    name: str
    interface: str
    batch_size: int = 1
    batching: bool = True


@configure
class ResourceSpec:
    """Declaration for one managed resource.

    A spec describes what raw resource to construct, which runtime target should
    own it, and which other resources should be injected into its constructor.

    :param name: Unique resource name within a manager.
    :param resource_name: Registry key identifying the resource entry. Concrete
        config instances may omit this and rely on registry reverse lookup.
    :param runtime: Runtime target name. ``None`` uses the registry default.
    :param config: Raw resource config object or a dict materialized through the
        entry config class.
    :param refs: Mapping from constructor parameter name to resource name.
    :param runtime_options: Runtime deployment and scheduling options. These are
        validated by ``ResourceManager`` according to the selected runtime.
    """

    name: str
    resource_name: str | None = None
    runtime: str | None = None
    config: Any = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)
    runtime_options: dict[str, Any] = field(default_factory=dict)


@configure
class ResourcesConfig:
    """Resource graph declaration consumed by ``ResourceManager.load``.

    ``ResourcesConfig`` describes the resources available to one manager and
    which of them should be eagerly loaded. Resources not listed in ``preload``
    remain lazy and are instantiated on first ``get()``.

    :param resources: Resource declarations keyed by unique ``name``.
    :param preload: Resource names to load eagerly when creating a manager.
    """

    resources: list[ResourceSpec] = field(default_factory=list)
    preload: list[str] = field(default_factory=list)
