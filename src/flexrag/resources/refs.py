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
class RuntimeConfig:
    """Runtime selection and execution overrides for a resource.

    ``None`` values leave the corresponding setting to the selected runtime
    and resource entry defaults. Runtime-specific combinations are validated by
    ``ResourceManager`` after the runtime name has been resolved.

    :param name: Runtime target name. ``None`` uses the resource entry default.
    :param batch_size: Optional public-call batch size override.
    :param max_concurrency: Optional async/process primitive-call concurrency
        limit. Direct resources are always serialized.
    :param rpm: Optional attempt-level requests-per-minute limit.
    :param worker_count: Optional process worker count.
    :param device_groups: Optional process worker accelerator placement.
    :param retry_times: Optional async runtime retry count.
    :param retry_min_delay: Optional minimum retry delay in seconds.
    :param retry_max_delay: Optional maximum retry delay in seconds.
    :param timeout: Optional async runtime per-attempt timeout in seconds.
    """

    name: str | None = None
    batch_size: int | None = None
    max_concurrency: int | None = None
    rpm: float | None = None
    worker_count: int | None = None
    device_groups: list[list[str]] | None = None
    retry_times: int | None = None
    retry_min_delay: float | None = None
    retry_max_delay: float | None = None
    timeout: float | None = None


@configure
class ResourceSpec:
    """Declaration for one managed resource.

    A spec describes what raw resource to construct, which runtime target should
    own it, and which other resources should be injected into its constructor.

    :param name: Unique resource name within a manager.
    :param resource_name: Registry key identifying the resource entry. Concrete
        resource config instances may omit this and rely on registry reverse
        lookup.
    :param resource_config: Raw resource config object or a dict materialized
        through the entry config class.
    :param runtime_config: Runtime selection and execution overrides.
    :param refs: Mapping from constructor parameter name to one resource name or
        a named, one-level mapping of resource names.
    """

    name: str
    resource_name: str | None = None
    resource_config: Any = field(default_factory=dict)
    runtime_config: RuntimeConfig = field(default_factory=RuntimeConfig)
    refs: dict[str, str | dict[str, str]] = field(default_factory=dict)


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
