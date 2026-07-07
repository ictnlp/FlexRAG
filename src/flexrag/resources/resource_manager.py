import inspect
from dataclasses import field
from typing import Any

from flexrag.common import configure

from .handles import (
    ChunkerHandle,
    ContextStoreHandle,
    EncoderHandle,
    GeneratorHandle,
    RankerHandle,
    RefinerHandle,
    RuntimeHandleBase,
    ScorerHandle,
    TokenizerHandle,
)
from .invocations import (
    BatchGeneratorInvocation,
    ChunkerInvocation,
    ContextStoreInvocation,
    EncoderInvocation,
    RankerInvocation,
    RefinerInvocation,
    ScorerInvocation,
    SingleSampleGeneratorInvocation,
    TokenizerInvocation,
)
from .registry import ResourceEntry, Resources
from .runtime_adapters import (
    DirectRuntimeAdapter,
    ProcessRuntimeAdapter,
    RemoteRuntimeAdapter,
)

_INTERFACE_HANDLE_TYPES: dict[str, type[RuntimeHandleBase]] = {
    "encoder": EncoderHandle,
    "generator": GeneratorHandle,
    "scorer": ScorerHandle,
    "ranker": RankerHandle,
    "refiner": RefinerHandle,
    "chunker": ChunkerHandle,
    "context_store": ContextStoreHandle,
    "tokenizer": TokenizerHandle,
}

_COMPOSITIONS: dict[tuple[str, type[Any]], tuple[type[Any], dict[str, Any]]] = {
    ("encoder", ProcessRuntimeAdapter): (
        EncoderInvocation,
        {"batch_method": "_encode_batch", "batch_size": 32},
    ),
    ("encoder", RemoteRuntimeAdapter): (
        EncoderInvocation,
        {"batch_method": "_async_encode_batch", "batch_size": 32},
    ),
    ("generator", ProcessRuntimeAdapter): (
        BatchGeneratorInvocation,
        {
            "generate_method": "_generate_batch",
            "chat_method": "_chat_batch",
            "batch_size": 1,
        },
    ),
    ("generator", RemoteRuntimeAdapter): (
        SingleSampleGeneratorInvocation,
        {
            "generate_method": "_async_generate_one",
            "chat_method": "_async_chat_one",
        },
    ),
    ("scorer", ProcessRuntimeAdapter): (
        ScorerInvocation,
        {"score_method": "_score_batch", "batch_size": 32},
    ),
    ("ranker", DirectRuntimeAdapter): (RankerInvocation, {}),
    ("ranker", RemoteRuntimeAdapter): (RankerInvocation, {}),
    ("refiner", DirectRuntimeAdapter): (RefinerInvocation, {}),
    ("chunker", DirectRuntimeAdapter): (ChunkerInvocation, {}),
    ("context_store", DirectRuntimeAdapter): (ContextStoreInvocation, {}),
    ("tokenizer", DirectRuntimeAdapter): (TokenizerInvocation, {}),
}


@configure
class ResourceSpec:
    """Declaration for one named runtime resource.

    :param name: Globally unique resource name.
    :param config: Concrete resource configuration. The config class must be
        registered in ``Resources``. When loading from serialized data, this may
        be a dictionary and ``resource`` must name the registered resource used
        to construct the concrete config.
    :param resource: Serialized resource discriminator. Python callers may omit
        this when ``config`` is already a concrete config instance; it will be
        filled with the registered canonical short name.
    :param runtime_kwargs: Runtime adapter constructor keyword arguments.
    :param invocation_kwargs: Invocation constructor keyword arguments. Use this
        for interface-call policy such as deployment batch size.
    :param refs: Constructor dependencies expressed as resource-name
        references. Each value must be the name of another resource managed by
        the same manager. Defaults to an empty dict.
    """

    name: str
    config: Any
    resource: str | None = None
    runtime_kwargs: dict[str, Any] = field(default_factory=dict)
    invocation_kwargs: dict[str, Any] = field(default_factory=dict)
    refs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalize serialized configs into concrete resource configs."""
        if isinstance(self.config, dict):
            if self.resource is None:
                raise ValueError(
                    "resource is required when config is loaded from a dictionary."
                )
            entry = Resources.resolve_name(self.resource)
            self.config = entry.config_class(**self.config)
            return

        entry = Resources.resolve(self.config)
        if self.resource is None:
            self.resource = entry.short_names[0]
            return
        if Resources.resolve_name(self.resource) is not entry:
            raise ValueError(
                f"resource {self.resource!r} does not match config class "
                f"{type(self.config)!r}."
            )
        return


@configure
class ResourceManagerConfig:
    """Configuration for named runtime resources.

    :param resources: Runtime resource declarations. Resource names must be
        globally unique. Defaults to an empty list.
    :param preload: Resource names to instantiate when
        :meth:`ResourceManager.load` is called. Resources not listed here are
        loaded lazily on first access. Defaults to an empty list.
    """

    resources: list[ResourceSpec] = field(default_factory=list)
    preload: list[str] = field(default_factory=list)


class ResourceManager:
    """Load, cache, and close named runtime resources.

    ``ResourceManager`` owns the runtime resources declared by
    :class:`ResourceManagerConfig`. Resources are resolved through the global
    ``Resources`` metadata register, loaded lazily by default, exposed through
    typed handles, cached by globally unique resource name, and closed in reverse
    load order.
    """

    def __init__(self, cfg: ResourceManagerConfig):
        """Initialize the manager without loading any resources.

        Use :meth:`load` when ``cfg.preload`` should be honored.

        :param cfg: The named resource configuration.
        """
        self.cfg = cfg
        self._specs = self._build_specs(cfg.resources)
        self._resources: dict[str, Any] = {}
        self._invocations: dict[str, Any] = {}
        self._handles: dict[str, RuntimeHandleBase] = {}
        self._load_order: list[str] = []
        self._closed = False
        return

    @staticmethod
    def _build_specs(resources: list[ResourceSpec]) -> dict[str, ResourceSpec]:
        specs: dict[str, ResourceSpec] = {}
        for spec in resources:
            if spec.name in specs:
                raise ValueError(f"Duplicate resource name: {spec.name}")
            specs[spec.name] = spec
        return specs

    @classmethod
    def load(cls, cfg: ResourceManagerConfig) -> "ResourceManager":
        """Create a manager and instantiate resources listed in ``cfg.preload``.

        Resources not listed in ``cfg.preload`` remain unloaded until
        :meth:`get` requests them.

        :param cfg: The named resource configuration.
        :return: A resource manager bound to ``cfg``.
        """
        manager = cls(cfg)
        for ref in cfg.preload:
            manager.get(ref)
        return manager

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed.")
        return

    def _get_spec(self, name: str) -> ResourceSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Resource not found: {name}") from exc

    def _resolve_refs(self, spec: ResourceSpec) -> dict[str, Any]:
        refs: dict[str, Any] = {}
        for param_name, resource_name in spec.refs.items():
            if not isinstance(resource_name, str):
                raise TypeError(
                    f"Resource ref {param_name!r} for {spec.name!r} must be a "
                    "resource name string."
                )
            refs[param_name] = self.get(resource_name)
        return refs

    def _merge_constructor_kwargs(
        self,
        spec: ResourceSpec,
        entry: ResourceEntry,
    ) -> dict[str, Any]:
        if spec.refs and not issubclass(entry.runtime_adapter_cls, DirectRuntimeAdapter):
            raise ValueError(
                f"Resource {spec.name!r} uses refs, but only direct runtime "
                "resources can receive dependency refs."
            )
        conflicts = spec.refs.keys() & spec.runtime_kwargs.keys()
        if conflicts:
            conflict_names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"Resource {spec.name!r} has duplicate constructor kwargs from "
                f"refs and runtime_kwargs: {conflict_names}."
            )
        refs = self._resolve_refs(spec)
        return {**refs, **spec.runtime_kwargs}

    @staticmethod
    def _get_handle_cls(interface: str) -> type[RuntimeHandleBase]:
        try:
            return _INTERFACE_HANDLE_TYPES[interface]
        except KeyError as exc:
            raise TypeError(
                f"Resource interface is not supported by ResourceManager: "
                f"{interface!r}."
            ) from exc

    @staticmethod
    def _get_composition(
        interface: str,
        runtime_adapter_cls: type[Any],
    ) -> tuple[type[Any], dict[str, Any]]:
        try:
            return _COMPOSITIONS[(interface, runtime_adapter_cls)]
        except KeyError:
            pass

        for (candidate_interface, candidate_runtime_cls), composition in (
            _COMPOSITIONS.items()
        ):
            if candidate_interface == interface and issubclass(
                runtime_adapter_cls,
                candidate_runtime_cls,
            ):
                return composition
        raise TypeError(
            "Resource runtime composition is not supported by "
            f"ResourceManager: interface={interface!r}, "
            f"runtime_adapter_cls={runtime_adapter_cls!r}."
        )

    def _load_resource(self, spec: ResourceSpec, entry: ResourceEntry) -> Any:
        constructor_kwargs = self._merge_constructor_kwargs(spec, entry)
        return entry.runtime_adapter_cls(
            spec.config,
            impl_cls=entry.impl_cls,
            **constructor_kwargs,
        )

    def _build_invocation(
        self,
        spec: ResourceSpec,
        entry: ResourceEntry,
        resource: Any,
    ) -> Any:
        invocation_cls, defaults = self._get_composition(
            entry.interface,
            entry.runtime_adapter_cls,
        )
        invocation_kwargs = {**defaults, **spec.invocation_kwargs}
        return invocation_cls(resource, **invocation_kwargs)

    def get(self, name: str) -> Any:
        """Get a named resource, loading and caching it on first access.

        :param name: The globally unique resource name.
        :raises RuntimeError: If the manager has already been closed.
        :raises ValueError: If refs and runtime kwargs conflict.
        :raises KeyError: If the resource name is not declared.
        :raises KeyError: If the concrete resource config is not registered.
        :raises TypeError: If the selected resource interface has no handle mapping.
        :return: The loaded runtime handle.
        """
        self._ensure_not_closed()
        if name not in self._resources:
            spec = self._get_spec(name)
            entry = Resources.resolve(spec.config)
            handle_cls = self._get_handle_cls(entry.interface)
            self._resources[name] = self._load_resource(spec, entry)
            self._invocations[name] = self._build_invocation(
                spec,
                entry,
                self._resources[name],
            )
            self._load_order.append(name)
            self._handles[name] = handle_cls(self._invocations[name])
        if name not in self._handles:
            spec = self._get_spec(name)
            entry = Resources.resolve(spec.config)
            handle_cls = self._get_handle_cls(entry.interface)
            if name not in self._invocations:
                self._invocations[name] = self._build_invocation(
                    spec,
                    entry,
                    self._resources[name],
                )
            self._handles[name] = handle_cls(self._invocations[name])
        return self._handles[name]

    async def _aclose_resource(self, resource) -> None:
        aclose = getattr(resource, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
            return

        close = getattr(resource, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        return

    async def aclose(self) -> None:
        """Asynchronously close all loaded resources.

        Resources are closed in reverse load order. For each resource, ``aclose`` is
        preferred when available; otherwise ``close`` is used. Calling this method
        multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True
        self._handles.clear()
        self._invocations.clear()
        while self._load_order:
            key = self._load_order.pop()
            resource = self._resources.pop(key, None)
            if resource is not None:
                await self._aclose_resource(resource)
        return

    def close(self) -> None:
        """Synchronously close all loaded resources.

        Resources are closed in reverse load order using their ``close`` method when
        present. Calling this method multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True
        self._handles.clear()
        self._invocations.clear()
        while self._load_order:
            key = self._load_order.pop()
            resource = self._resources.pop(key, None)
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        return

    def __enter__(self) -> "ResourceManager":
        self._ensure_not_closed()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
        return

    async def __aenter__(self) -> "ResourceManager":
        self._ensure_not_closed()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()
        return
