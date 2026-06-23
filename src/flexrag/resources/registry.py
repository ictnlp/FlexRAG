from dataclasses import dataclass
from typing import Any, TypeVar

ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class ResourceEntry:
    """Metadata for a registered resource implementation.

    ``ResourceEntry`` is intentionally descriptive only. It records how a
    concrete config maps to a raw implementation and its default runtime
    adapter, but it does not instantiate either object.

    :param short_names: Serialized discriminator names for this resource.
    :param interface: Resource call interface used by resource managers to
        select the returned handle.
    :param config_class: Concrete configuration class for the raw resource.
    :param impl_cls: Raw resource implementation class.
    :param runtime_adapter_cls: Runtime adapter class used by resource managers.
    """

    short_names: tuple[str, ...]
    interface: str
    config_class: type[Any]
    impl_cls: type[Any]
    runtime_adapter_cls: type[Any]


class _ResourceRegister:
    """Lightweight registry for resource metadata.

    The register resolves concrete config classes or serialized short names to
    :class:`ResourceEntry` objects. It deliberately does not load resources,
    create runtime adapters, wrap handles, or manage lifecycle.
    """

    def __init__(self) -> None:
        """Initialize an empty resource metadata register."""
        self._entries_by_config_class: dict[type[Any], ResourceEntry] = {}
        self._entries_by_short_name: dict[str, ResourceEntry] = {}
        return

    @property
    def entries(self) -> tuple[ResourceEntry, ...]:
        """Return all registered entries in registration order.

        :return: Registered resource entries.
        """
        return tuple(self._entries_by_config_class.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return all registered serialized discriminator names.

        :return: Registered short names.
        """
        return tuple(self._entries_by_short_name)

    def _validate_short_names(self, short_names: tuple[str, ...]) -> None:
        if not short_names:
            raise ValueError("At least one short name is required.")
        seen: set[str] = set()
        for name in short_names:
            if not isinstance(name, str):
                raise TypeError("Resource short names must be strings.")
            if not name:
                raise ValueError("Resource short names must not be empty.")
            if name in seen:
                raise ValueError(f"Duplicate short name in one registration: {name}")
            seen.add(name)
        return

    def register(
        self,
        *short_names: str,
        interface: str,
        config_class: type[ConfigT],
        runtime_adapter_cls: type[Any],
    ):
        """Register a resource implementation.

        :param short_names: One or more serialized discriminator names.
        :param interface: Resource call interface used to select a handle.
        :param config_class: Concrete configuration class for the resource.
        :param runtime_adapter_cls: Runtime adapter class selected by default.
        :raises ValueError: If names or config classes conflict.
        :return: A decorator that records the implementation class unchanged.
        """
        self._validate_short_names(short_names)
        if not isinstance(interface, str):
            raise TypeError("interface must be a string.")
        if not interface:
            raise ValueError("interface must not be empty.")
        if not isinstance(config_class, type):
            raise TypeError("config_class must be a class.")
        if not isinstance(runtime_adapter_cls, type):
            raise TypeError("runtime_adapter_cls must be a class.")
        if config_class in self._entries_by_config_class:
            raise ValueError(f"Config class already registered: {config_class!r}")
        for name in short_names:
            if name in self._entries_by_short_name:
                raise ValueError(f"Resource short name already registered: {name}")

        def register_impl(impl_cls: type[Any]) -> type[Any]:
            if not isinstance(impl_cls, type):
                raise TypeError("Resource implementation must be a class.")
            entry = ResourceEntry(
                short_names=short_names,
                interface=interface,
                config_class=config_class,
                impl_cls=impl_cls,
                runtime_adapter_cls=runtime_adapter_cls,
            )
            self._entries_by_config_class[config_class] = entry
            for name in short_names:
                self._entries_by_short_name[name] = entry
            return impl_cls

        return register_impl

    def resolve(self, config: Any) -> ResourceEntry:
        """Resolve a resource entry from a concrete config instance.

        :param config: Concrete resource configuration instance.
        :raises KeyError: If the config class is not registered.
        :return: Registered metadata entry.
        """
        return self.resolve_config_class(type(config))

    def resolve_config_class(self, config_class: type[Any]) -> ResourceEntry:
        """Resolve a resource entry from a concrete config class.

        :param config_class: Concrete resource configuration class.
        :raises KeyError: If the config class is not registered.
        :return: Registered metadata entry.
        """
        try:
            return self._entries_by_config_class[config_class]
        except KeyError as exc:
            raise KeyError(f"Resource config class is not registered: {config_class!r}") from exc

    def resolve_name(self, short_name: str) -> ResourceEntry:
        """Resolve a resource entry from a serialized short name.

        :param short_name: Serialized discriminator name.
        :raises KeyError: If the name is not registered.
        :return: Registered metadata entry.
        """
        try:
            return self._entries_by_short_name[short_name]
        except KeyError as exc:
            raise KeyError(f"Resource short name is not registered: {short_name!r}") from exc

Resources = _ResourceRegister()


__all__ = ["ResourceEntry", "Resources"]
