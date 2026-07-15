import keyword
import types
from dataclasses import asdict, field, fields, is_dataclass
from pathlib import Path
from typing import Annotated, Callable, Generic, Optional, TypeVar, dataclass_transform

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic.dataclasses import ConfigDict, Field, FieldInfo, dataclass

T = TypeVar("T")


def extract_config(config, config_cls: type[T]) -> T:
    """Extracts the configuration from a pydantic dataclass, omegaconf.DictConfig or dict.

    :param config: The configuration source; can be a ``DictConfig``, dict, or dataclass instance.
    :param config_cls: The target pydantic dataclass type.
    :type config_cls: type[T]
    :return: An instance of *config_cls* populated with the extracted values.
    :rtype: T
    :raises TypeError: If *config* is not a supported type.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        config = config_cls(**config)
    elif is_dataclass(config):
        field_names = {f.name for f in fields(config_cls)}
        kwargs = {name: getattr(config, name) for name in field_names}
        config = config_cls(**kwargs)
    else:
        raise TypeError(f"Expected {config_cls}, got {type(config)}")
    return config


def make_dataclass(
    cls_name,
    fields,
    *,
    bases=(),
    namespace=None,
    repr=True,
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only=False,
    slots=False,
    config=None,
    validate_on_init=None,
):
    """Return a new dynamically created pydantic dataclass."""

    if namespace is None:
        namespace = {}

    # While we're looking through the field names, validate that they
    # are identifiers, are not keywords, and not duplicates.
    seen = set()
    annotations = {}
    defaults = {}
    for item in fields:
        if isinstance(item, str):
            name = item
            tp = "typing.Any"
        elif len(item) == 2:
            (
                name,
                tp,
            ) = item
        elif len(item) == 3:
            name, tp, spec = item
            defaults[name] = spec
        else:
            raise TypeError(f"Invalid field: {item!r}")

        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f"Field names must be valid identifiers: {name!r}")
        if keyword.iskeyword(name):
            raise TypeError(f"Field names must not be keywords: {name!r}")
        if name in seen:
            raise TypeError(f"Field name duplicated: {name!r}")

        seen.add(name)
        annotations[name] = tp

    # Update 'ns' with the user-supplied namespace plus our calculated values.
    def exec_body_callback(ns):
        ns.update(namespace)
        ns.update(defaults)
        ns["__annotations__"] = annotations

    # We use `types.new_class()` instead of simply `type()` to allow dynamic creation
    # of generic dataclasses.
    cls = types.new_class(cls_name, bases, {}, exec_body_callback)

    # Apply the normal decorator.
    return dataclass(
        cls,
        init=False,  # pydantic dataclass only supports `init=False`
        repr=repr,
        eq=eq,
        order=order,
        unsafe_hash=unsafe_hash,
        frozen=frozen,
        kw_only=kw_only,
        slots=slots,
        config=config,
        validate_on_init=validate_on_init,
    )


def Choices(*args: str) -> FieldInfo:
    """Create a pydantic Field constrained to the given choices.

    This is useful as hydra-core does not support ``Literal`` types.

    :param args: The allowed choice strings.
    :return: A pydantic ``FieldInfo`` with a regex constraint.
    :rtype: FieldInfo
    """
    choices = list(args)
    pattern = f"^({'|'.join(choices)})$"
    return Field(pattern=pattern)


_T = TypeVar("_T")


@dataclass_transform()
def _create_pydantic_dataclass(config: ConfigDict) -> Callable[[type[_T]], type[_T]]:
    def decorator(cls: type[_T] = None, *, frozen=False, kw_only=False) -> type[_T]:
        if cls is None:
            return lambda cls: decorator(cls, frozen=frozen, kw_only=kw_only)

        cls = dataclass(config=config, frozen=frozen, kw_only=kw_only)(cls)

        def dumps(self) -> str:
            """Dump the dataclass to a YAML string."""
            return yaml.safe_dump(asdict(self))

        def dump(self, path: str | Path):
            """Dump the dataclass to a YAML file."""
            path = Path(path)
            path.write_text(self.dumps(), encoding="utf-8")

        @classmethod
        def loads(cls, s: str) -> _T:
            """Load the dataclass from a YAML string."""
            data = yaml.safe_load(s)
            if not isinstance(data, dict):
                raise ValueError("YAML string must represent a dictionary.")
            return cls(**data)

        @classmethod
        def load(cls, path: str | Path) -> _T:
            """Load the dataclass from a YAML file."""
            path = Path(path)
            return cls.loads(path.read_text(encoding="utf-8"))

        setattr(cls, "dumps", dumps)
        setattr(cls, "dump", dump)
        setattr(cls, "loads", loads)
        setattr(cls, "load", load)
        return cls

    return decorator


# These two variables are intended as a shortcut
# for creating pydantic.dataclasses.dataclass instances.
configure = _create_pydantic_dataclass(
    ConfigDict(extra="forbid", validate_assignment=True)
)
data = _create_pydantic_dataclass(
    ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)
)

RegistedType = TypeVar("RegistedType")


class Register(Generic[RegistedType]):
    def __init__(self, register_name: str):
        """Initialize the register.

        :param register_name: The name of the register.
        """
        self.name = register_name
        self._items = {}
        self._shortcuts = {}

    def __call__(self, *short_names: str, config_class=None):
        """Register an item to the register.

        :param short_names: The short names of the item.
        :type short_names: str
        :param config_class: The config class of the item, defaults to None.
        :type config_class: dataclass
        :return: The item.
        :rtype: Any
        """

        def register_item(item):
            main_name = str(item).split(".")[-1][:-2]
            # check name conflict
            assert main_name not in self._items, f"Name Conflict {main_name}"
            assert main_name not in self._shortcuts, f"Name Conflict {main_name}"
            for name in short_names:
                assert name not in self._items, f"Name Conflict {name}"
                assert name not in self._shortcuts, f"Name Conflict {name}"

            # register the item
            self._items[main_name] = {
                "item": item,
                "main_name": main_name,
                "short_names": short_names,
                "config_class": config_class,
            }
            for name in short_names:
                self._shortcuts[name] = main_name
            return item

        return register_item

    @property
    def names(self) -> list[str]:
        """Get the names of the registered items."""
        return list(self._items.keys()) + list(self._shortcuts.keys())

    @property
    def mainnames(self) -> list[str]:
        """Get the main names of the registered items."""
        return list(self._items.keys())

    def __getitem__(self, key: str) -> dict:
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]

    def make_config(
        self,
        allow_multiple: bool = False,
        default: Optional[str] = None,
        config_name: Optional[str] = None,
    ):
        """Make a config class for the registered items.

        :param allow_multiple: Whether to allow multiple items to be selected, defaults to False.
        :type allow_multiple: bool, optional
        :param default: The default item to select, defaults to None.
        :type default: Optional[str], optional
        :param config_name: The name of the config class, defaults to None.
        :type config_name: str, optional
        :return: The config class.
        :rtype: dataclass
        """
        choice_name = f"{self.name}_type"
        config_name = f"{self.name}_config" if config_name is None else config_name
        if allow_multiple:
            config_fields = [
                (
                    choice_name,
                    list[Annotated[str, Choices(*self.names)]],
                    field(default_factory=list),
                )
            ]
        else:
            config_fields = [
                (
                    choice_name,
                    Optional[Annotated[str, Choices(*self.names)]],
                    field(default=default),
                )
            ]
        config_fields += [
            (
                f"{self[name]['short_names'][0]}_config",
                Optional[self[name]["config_class"]],
                field(default_factory=self._items[name]["config_class"]),
            )
            for name in self.mainnames
            if self[name]["config_class"] is not None
        ]
        generated_config = make_dataclass(config_name, config_fields)

        # set docstring
        docstring = (
            f"Configuration class for {self.name} "
            f"(name: {config_name}, default: {default}).\n\n"
        )
        docstring += f":param {choice_name}: The {self.name} type to use.\n"
        if allow_multiple:
            docstring += f":type {choice_name}: list[str]\n"
        else:
            docstring += f":type {choice_name}: str\n"
        for name in self.mainnames:
            if self[name]["config_class"] is not None:
                docstring += f":param {self[name]['short_names'][0]}_config: The config for {name}.\n"
                docstring += f":type {self[name]['short_names'][0]}_config: {self[name]['config_class'].__name__}\n"
        generated_config.__doc__ = docstring
        return generated_config

    def load(
        self,
        config: DictConfig,
        **kwargs,
    ) -> RegistedType | list[RegistedType]:
        """Load the item(s) from the generated config.

        :param config: The config generated by `make_config` method.
        :type config: DictConfig
        :param kwargs: The additional arguments to pass to the item(s).
        :type kwargs: Any
        :raises ValueError: If the item type is invalid.
        :return: The loaded item(s).
        :rtype: RegistedType | list[RegistedType]
        """

        def load_item(type_str: str) -> RegistedType:
            if type_str not in self:
                raise ValueError(f"Invalid {self.name} type: {type_str}")
            cfg_name = f"{self[type_str]['short_names'][0]}_config"
            sub_cfg = getattr(config, cfg_name, None)
            if sub_cfg is None:
                return self[type_str]["item"](**kwargs)
            return self[type_str]["item"](sub_cfg, **kwargs)

        choice = getattr(config, f"{self.name}_type", None)
        if choice is None:
            return None
        if isinstance(choice, (list, ListConfig)):
            loaded = []
            for name in choice:
                loaded.append(load_item(str(name)))
            return loaded
        return load_item(str(choice))

    def __contains__(self, key: str) -> bool:
        return key in self.names
