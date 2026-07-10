from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .entry import ResourceEntry
from .handles import HANDLE_TYPES, TypedHandle
from .refs import ResourceRefDescriptor, ResourcesConfig, ResourceSpec
from .registry import Resources, _ResourceRegister
from .runtime import AsyncTarget, DirectTarget, ProcessTarget, RuntimeTarget
from .runtime.placement import worker_env_updates_from_device_groups


@dataclass(frozen=True)
class RuntimeSettings:
    """Validated runtime settings for one managed resource.

    This is an internal normalized form of ``ResourceSpec.runtime_options``.
    Common options are ``batch_size``, ``max_concurrency``, and ``rpm``.
    Process-only options are worker count and per-worker environment updates.
    Async-only options are retry and timeout settings.
    """

    batch_size: int
    max_concurrency: int
    rpm: float
    worker_count: int = 1
    worker_env_updates: tuple[dict[str, str], ...] | None = None
    retry_times: int = 0
    retry_min_delay: float = 1.0
    retry_max_delay: float = 60.0
    timeout: float = 0.0


class ResourceManager:
    """Lazy resource graph manager.

    The manager resolves specs through a registry, constructs targets lazily on
    first ``get()``, returns typed handles, caches loaded handles, and owns all
    target lifecycle. Handles returned by the manager do not expose close
    methods; closing the manager closes loaded targets in reverse load order.
    """

    def __init__(
        self,
        specs: list[ResourceSpec],
        *,
        registry: _ResourceRegister | None = None,
    ) -> None:
        """Create a manager for a fixed set of resource specs.

        :param specs: Resource declarations keyed by unique ``name``.
        :param registry: Optional resource registry. The global formal registry
            is used when omitted.
        :raises ValueError: If resource names are duplicated.
        """
        self._specs = self._build_specs(specs)
        self._registry = Resources if registry is None else registry
        self._targets: dict[str, RuntimeTarget] = {}
        self._handles: dict[str, TypedHandle] = {}
        self._load_order: list[str] = []
        self._closed = False
        return

    @staticmethod
    def _build_specs(specs: list[ResourceSpec]) -> dict[str, ResourceSpec]:
        by_name = {}
        for spec in specs:
            if spec.name in by_name:
                raise ValueError(f"Duplicate resource name: {spec.name}")
            by_name[spec.name] = spec
        return by_name

    @classmethod
    def load(
        cls,
        config: ResourcesConfig,
        *,
        registry: _ResourceRegister | None = None,
    ) -> ResourceManager:
        """Create a manager and eagerly load configured resources.

        :param config: Resource graph declaration.
        :param registry: Optional resource registry. The global formal registry
            is used when omitted.
        Transitive refs of each preload root are loaded before the root itself
        so close order still follows dependency ownership.

        :returns: Resource manager with ``config.preload`` resource graphs loaded.
        :raises KeyError: If a preload resource name is not declared.
        :raises ValueError: If preloaded refs contain a cycle.
        """
        manager = cls(config.resources, registry=registry)
        loaded: set[str] = set()

        def preload(resource_name: str, visiting: tuple[str, ...]) -> None:
            if resource_name in loaded:
                return
            if resource_name in visiting:
                cycle = " -> ".join((*visiting, resource_name))
                raise ValueError(f"Resource preload refs contain a cycle: {cycle}")
            spec = manager._get_spec(resource_name)
            next_visiting = (*visiting, resource_name)
            for ref_name in manager._validated_refs(spec).values():
                preload(ref_name, next_visiting)
            manager.get(resource_name)
            loaded.add(resource_name)
            return

        try:
            for resource_name in config.preload:
                preload(resource_name, ())
        except Exception:
            manager.close()
            raise
        return manager

    def get(self, name: str) -> TypedHandle:
        """Load or return a managed resource handle.

        Loading is lazy. The selected runtime target is created, refs are
        resolved according to runtime requirements, and the typed handle is
        cached for subsequent calls.

        :param name: Resource name declared in this manager.
        :returns: Typed handle for the resource interface.
        :raises KeyError: If the resource name or type is unknown.
        :raises TypeError: If the resource interface has no handle mapping.
        :raises ValueError: If runtime options are invalid.
        :raises RuntimeError: If the manager has already been closed.
        """
        self._ensure_not_closed()
        if name in self._handles:
            return self._handles[name]
        spec = self._get_spec(name)
        entry = self._resolve_entry(spec)
        runtime = spec.runtime or entry.default_runtime
        runtime_settings = self._runtime_settings(spec, entry, runtime)
        config = self._materialize_config(spec, entry)

        if runtime == "direct":
            target = DirectTarget(
                entry.raw_cls,
                config,
                refs=self._resolve_direct_refs(spec),
                batch_size=runtime_settings.batch_size,
                max_concurrency=runtime_settings.max_concurrency,
                rpm=runtime_settings.rpm,
            )
        elif runtime == "process":
            target = ProcessTarget(
                entry.raw_cls,
                config,
                refs=self._resolve_process_refs(spec),
                manager=self,
                worker_count=runtime_settings.worker_count,
                worker_env_updates=runtime_settings.worker_env_updates,
                batch_size=runtime_settings.batch_size,
                max_concurrency=runtime_settings.max_concurrency,
                rpm=runtime_settings.rpm,
            )
        elif runtime == "async":
            target = AsyncTarget(
                entry.raw_cls,
                config,
                refs=self._resolve_direct_refs(spec),
                batch_size=runtime_settings.batch_size,
                max_concurrency=runtime_settings.max_concurrency,
                rpm=runtime_settings.rpm,
                retry_times=runtime_settings.retry_times,
                retry_min_delay=runtime_settings.retry_min_delay,
                retry_max_delay=runtime_settings.retry_max_delay,
                timeout=runtime_settings.timeout,
            )
        else:
            raise ValueError(f"Unsupported runtime: {runtime}")

        handle_cls = self._get_handle_cls(entry)
        handle = handle_cls(
            target,
            batching=entry.batching,
        )
        self._targets[name] = target
        self._handles[name] = handle
        self._load_order.append(name)
        return handle

    def _ensure_not_closed(self) -> None:
        if self._closed:
            raise RuntimeError(f"{self.__class__.__name__} has been closed.")
        return

    def _get_spec(self, name: str) -> ResourceSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Resource not found: {name}") from exc

    def _resolve_entry(self, spec: ResourceSpec) -> ResourceEntry:
        if spec.resource_name is None:
            if isinstance(spec.config, dict):
                raise ValueError(
                    "resource_name is required when config is loaded from a "
                    "dictionary."
                )
            return self._registry.resolve_config(spec.config)

        entry = self._registry.resolve_name(spec.resource_name)
        if not isinstance(spec.config, dict) and entry.config_class is not None:
            config_entry = self._registry.resolve_config(spec.config)
            if config_entry is not entry:
                raise ValueError(
                    f"resource_name {spec.resource_name!r} does not match config "
                    f"class {type(spec.config)!r}."
                )
        return entry

    @staticmethod
    def _get_handle_cls(entry: ResourceEntry) -> type[TypedHandle]:
        try:
            return HANDLE_TYPES[entry.interface]
        except KeyError as exc:
            raise TypeError(
                f"Resource interface is not supported: {entry.interface!r}."
            ) from exc

    @staticmethod
    def _materialize_config(spec: ResourceSpec, entry: ResourceEntry) -> Any:
        if entry.config_class is None:
            return spec.config
        if isinstance(spec.config, dict):
            return entry.config_class(**spec.config)
        return spec.config

    def _runtime_settings(
        self,
        spec: ResourceSpec,
        entry: ResourceEntry,
        runtime: str,
    ) -> RuntimeSettings:
        options = spec.runtime_options
        allowed = {"batch_size", "max_concurrency", "rpm"}
        if runtime == "process":
            allowed.update({"device_groups", "worker_count"})
        elif runtime == "async":
            allowed.update(
                {
                    "retry_max_delay",
                    "retry_min_delay",
                    "retry_times",
                    "timeout",
                }
            )
        elif runtime != "direct":
            raise ValueError(f"Unsupported runtime: {runtime}")
        unsupported = set(options) - allowed
        if unsupported:
            raise ValueError(
                f"Unsupported {runtime} runtime_options: {sorted(unsupported)}"
            )
        default_batch_size = 32 if entry.batching else 1
        batch_size = self._positive_int_option(
            options,
            "batch_size",
            default_batch_size,
        )
        if not entry.batching and batch_size > 1:
            raise ValueError(
                f"Resource {entry.resource_name!r} does not support batched public "
                "calls."
            )
        rpm = self._non_negative_float_option(options, "rpm", 0.0)
        if runtime == "direct":
            max_concurrency = self._positive_int_option(
                options,
                "max_concurrency",
                1,
            )
            return RuntimeSettings(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                rpm=rpm,
            )
        if runtime == "async":
            max_concurrency = self._positive_int_option(
                options,
                "max_concurrency",
                1,
            )
            retry_times = self._non_negative_int_option(
                options,
                "retry_times",
                0,
            )
            retry_min_delay = self._non_negative_float_option(
                options,
                "retry_min_delay",
                1.0,
            )
            retry_max_delay = self._non_negative_float_option(
                options,
                "retry_max_delay",
                60.0,
            )
            if retry_max_delay < retry_min_delay:
                raise ValueError(
                    "retry_max_delay must be greater than or equal to retry_min_delay."
                )
            timeout = self._non_negative_float_option(options, "timeout", 0.0)
            return RuntimeSettings(
                batch_size=batch_size,
                max_concurrency=max_concurrency,
                rpm=rpm,
                retry_times=retry_times,
                retry_min_delay=retry_min_delay,
                retry_max_delay=retry_max_delay,
                timeout=timeout,
            )

        if "device_groups" in options and "worker_count" in options:
            raise ValueError("worker_count and device_groups cannot be used together.")

        worker_env_updates = None
        if "device_groups" in options:
            worker_env_updates = worker_env_updates_from_device_groups(
                options["device_groups"]
            )
            worker_count = len(worker_env_updates)
        else:
            worker_count = self._positive_int_option(options, "worker_count", 1)
        if worker_count > 1 and not entry.parallel_safe:
            raise ValueError(
                f"Resource {entry.resource_name!r} is not safe to run with "
                "worker_count > 1."
            )
        max_concurrency = self._positive_int_option(
            options,
            "max_concurrency",
            worker_count,
        )
        if max_concurrency > worker_count:
            raise ValueError(
                "process max_concurrency must be less than or equal to worker_count."
            )
        return RuntimeSettings(
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            rpm=rpm,
            worker_count=worker_count,
            worker_env_updates=worker_env_updates,
        )

    @staticmethod
    def _positive_int_option(
        options: Mapping[str, Any],
        name: str,
        default: int,
    ) -> int:
        value = options.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be an integer greater than or equal to 1.")
        return value

    @staticmethod
    def _non_negative_float_option(
        options: Mapping[str, Any],
        name: str,
        default: float,
    ) -> float:
        value = options.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError(f"{name} must be a non-negative number.")
        return float(value)

    @staticmethod
    def _non_negative_int_option(
        options: Mapping[str, Any],
        name: str,
        default: int,
    ) -> int:
        value = options.get(name, default)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
        return value

    def _resolve_direct_refs(self, spec: ResourceSpec) -> dict[str, TypedHandle]:
        return {
            param_name: self.get(resource_name)
            for param_name, resource_name in self._validated_refs(spec).items()
        }

    def _resolve_process_refs(
        self,
        spec: ResourceSpec,
    ) -> dict[str, ResourceRefDescriptor]:
        descriptors = {}
        for param_name, resource_name in self._validated_refs(spec).items():
            ref_spec = self._get_spec(resource_name)
            entry = self._resolve_entry(ref_spec)
            runtime = ref_spec.runtime or entry.default_runtime
            runtime_settings = self._runtime_settings(ref_spec, entry, runtime)
            descriptors[param_name] = ResourceRefDescriptor(
                name=resource_name,
                interface=entry.interface,
                batch_size=runtime_settings.batch_size,
                batching=entry.batching,
            )
        return descriptors

    @staticmethod
    def _validated_refs(spec: ResourceSpec) -> dict[str, str]:
        refs: dict[str, str] = {}
        for param_name, resource_name in spec.refs.items():
            if not isinstance(resource_name, str):
                raise TypeError(
                    f"Resource ref {param_name!r} for {spec.name!r} must be a "
                    "resource name string."
                )
            refs[param_name] = resource_name
        return refs

    def close(self) -> None:
        """Synchronously close loaded targets and clear manager caches.

        Targets are closed in reverse load order. The manager attempts to close
        every loaded target even if one close call fails, clears cached handles
        and targets afterwards, and then re-raises the first close error.
        Calling this method multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        while self._load_order:
            name = self._load_order.pop()
            target = self._targets.pop(name, None)
            if target is None:
                continue
            try:
                target.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._handles.clear()
        self._targets.clear()
        if first_error is not None:
            raise first_error
        return

    async def async_close(self) -> None:
        """Asynchronously close loaded targets and clear manager caches.

        Targets are closed in reverse load order. The manager attempts to close
        every loaded target even if one close call fails, clears cached handles
        and targets afterwards, and then re-raises the first close error.
        Calling this method multiple times is safe.
        """
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        while self._load_order:
            name = self._load_order.pop()
            target = self._targets.pop(name, None)
            if target is None:
                continue
            try:
                await target.async_close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self._handles.clear()
        self._targets.clear()
        if first_error is not None:
            raise first_error
        return

    def __enter__(self) -> ResourceManager:
        """Return this manager for synchronous context-manager use."""
        self._ensure_not_closed()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Close the manager when leaving a synchronous context."""
        self.close()
        return

    async def __aenter__(self) -> ResourceManager:
        """Return this manager for asynchronous context-manager use."""
        self._ensure_not_closed()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """Asynchronously close the manager when leaving a context."""
        await self.async_close()
        return
