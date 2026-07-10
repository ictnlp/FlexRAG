from __future__ import annotations

from typing import Any, TypeVar

from .entry import ResourceEntry

RawResourceT = TypeVar("RawResourceT", bound=type[Any])

_VALID_RUNTIMES = {"direct", "process", "async"}


class _ResourceRegister:
    """Registry for resource metadata.

    The register is the public extension point for declaring new resource
    implementations. It resolves resources by their canonical name or by config
    class, but it does not instantiate raw resources, targets, or handles.
    """

    def __init__(self) -> None:
        """Create an empty resource register."""
        self._entries_by_name: dict[str, ResourceEntry] = {}
        self._entries_by_config_class: dict[type[Any], ResourceEntry] = {}
        return

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered canonical resource names."""
        return tuple(self._entries_by_name)

    @property
    def entries(self) -> tuple[ResourceEntry, ...]:
        """Return registered entries in registration order."""
        return tuple(self._entries_by_name.values())

    def register(
        self,
        resource_name: str,
        *,
        interface: str,
        config_class: type[Any] | None = None,
        default_runtime: str = "direct",
        parallel_safe: bool = False,
        batching: bool = True,
    ):
        """Register a raw resource class.

        :param resource_name: Canonical resource name used by ``ResourceSpec``.
        :param interface: Interface name used to choose the typed handle.
        :param config_class: Optional config class for dict materialization and
            config-instance reverse lookup.
        :param default_runtime: Runtime used when a spec omits ``runtime``.
        :param parallel_safe: Whether process runtime may replicate the raw
            resource across workers.
        :param batching: Whether public handle calls may batch multiple samples.
        :raises ValueError: If names, config classes, or runtime names conflict.
        :return: A decorator that records the implementation class unchanged.
        """
        self._validate_registration(
            resource_name,
            interface=interface,
            config_class=config_class,
            default_runtime=default_runtime,
        )

        def register_raw(raw_cls: RawResourceT) -> RawResourceT:
            if not isinstance(raw_cls, type):
                raise TypeError("raw resource must be a class.")
            entry = ResourceEntry(
                resource_name=resource_name,
                raw_cls=raw_cls,
                interface=interface,
                config_class=config_class,
                default_runtime=default_runtime,
                parallel_safe=parallel_safe,
                batching=batching,
            )
            self._entries_by_name[resource_name] = entry
            if config_class is not None:
                self._entries_by_config_class[config_class] = entry
            return raw_cls

        return register_raw

    def resolve_name(self, resource_name: str) -> ResourceEntry:
        """Resolve a resource entry by canonical name."""
        try:
            return self._entries_by_name[resource_name]
        except KeyError as exc:
            raise KeyError(f"Resource name is not registered: {resource_name!r}") from exc

    def resolve_config_class(self, config_class: type[Any]) -> ResourceEntry:
        """Resolve a resource entry by config class."""
        try:
            return self._entries_by_config_class[config_class]
        except KeyError as exc:
            raise KeyError(
                f"Resource config class is not registered: {config_class!r}"
            ) from exc

    def resolve_config(self, config: Any) -> ResourceEntry:
        """Resolve a resource entry from a concrete config instance."""
        return self.resolve_config_class(type(config))

    def _validate_registration(
        self,
        resource_name: str,
        *,
        interface: str,
        config_class: type[Any] | None,
        default_runtime: str,
    ) -> None:
        if not isinstance(resource_name, str):
            raise TypeError("resource_name must be a string.")
        if not resource_name:
            raise ValueError("resource_name must not be empty.")
        if resource_name in self._entries_by_name:
            raise ValueError(f"Resource name already registered: {resource_name}")
        if not isinstance(interface, str):
            raise TypeError("interface must be a string.")
        if not interface:
            raise ValueError("interface must not be empty.")
        if config_class is not None and not isinstance(config_class, type):
            raise TypeError("config_class must be a class or None.")
        if config_class is not None and config_class in self._entries_by_config_class:
            raise ValueError(f"Config class already registered: {config_class!r}")
        if default_runtime not in _VALID_RUNTIMES:
            raise ValueError(f"Unsupported default runtime: {default_runtime!r}")
        return


Resources = _ResourceRegister()


__all__ = ["ResourceEntry", "Resources", "_ResourceRegister"]
