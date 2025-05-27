import json
import keyword
import os
import types
from dataclasses import field, fields, is_dataclass
from typing import Annotated, Callable, Generic, Optional, Type, TypeVar

from huggingface_hub import HfApi
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict, Field
from pydantic.dataclasses import ConfigDict, dataclass

from .default_vars import FLEXRAG_CACHE_DIR

T = TypeVar("T")


def extract_config(config, config_cls: Type[T]) -> T:
    """
    Extracts the configuration from a pydantic dataclass.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, dict):
        return config_cls(**config)
    elif is_dataclass(config):
        field_names = {f.name for f in fields(config_cls)}
        kwargs = {name: getattr(config, name) for name in field_names}
        return config_cls(**kwargs)
    else:
        raise TypeError(f"Expected {config_cls}, got {type(config)}")


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


def Choices(*args: str) -> Field:
    """
    A shortcut for creating a pydantic Field with a regex pattern that matches one of the provided choices.
    This is useful as hydra-core does not support `Literal` types.
    """
    choices = list(args)
    pattern = f"^({'|'.join(choices)})$"
    return Field(pattern=pattern)


_T = TypeVar("_T")


def _create_pydantic_dataclass(config: ConfigDict) -> Callable[[Type[_T]], Type[_T]]:
    def decorator(cls: Type[_T]) -> Type[_T]:
        return dataclass(config=config)(cls)  # type: ignore

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
    def __init__(self, register_name: str = None, allow_load_from_repo: bool = False):
        """Initialize the register.

        :param register_name: The name of the register, defaults to None.
        :type register_name: str, optional
        :param allow_load_from_repo: Whether to allow loading items from the HuggingFace Hub, defaults to False.
        :type allow_load_from_repo: bool, optional
        """
        self.name = register_name
        self.allow_load_from_repo = allow_load_from_repo
        self._items = {}
        self._shortcuts = {}
        return

    def __call__(self, *short_names: str, config_class=None):
        """Register an item to the register.

        :param short_names: The short names of the item.
        :type short_names: str
        :param config_class: The config class of the item, defaults to None.
        :type config_class: dataclass
        :return: The item.
        :rtype: Any
        """

        def registe_item(item):
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

        return registe_item

    def __iter__(self):
        return self._items.__iter__()

    @property
    def names(self) -> list[str]:
        """Get the names of the registered items."""
        return list(self._items.keys()) + list(self._shortcuts.keys())

    @property
    def mainnames(self) -> list[str]:
        """Get the main names of the registered items."""
        return list(self._items.keys())

    @property
    def shortnames(self) -> list[str]:
        """Get the short names of the registered items."""
        return list(self._shortcuts.keys())

    def __getitem__(self, key: str) -> dict:
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]

    def get(self, key: str, default=None) -> dict:
        """Get the item dict by name.

        :param key: The name of the item.
        :type key: str
        :param default: The default value to return, defaults to None.
        :type default: Any
        :return: The item dict containing the item, main_name, short_names, and config_class.
        :rtype: dict
        """
        if key not in self._items:
            if key not in self._shortcuts:
                return default
            key = self._shortcuts[key]
        return self._items[key]

    def get_item(self, key: str):
        """Get the item by name.

        :param key: The name of the item.
        :type key: str
        :return: The item.
        :rtype: Any
        """
        if key not in self._items:
            key = self._shortcuts[key]
        return self._items[key]["item"]

    def make_config(
        self,
        allow_multiple: bool = False,
        default: Optional[str] = MISSING,
        config_name: str = None,
    ):
        """Make a config class for the registered items.

        :param allow_multiple: Whether to allow multiple items to be selected, defaults to False.
        :type allow_multiple: bool, optional
        :param default: The default item to select, defaults to MISSING(???).
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
                self[name]["config_class"],
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
            # Try to load the item from the HuggingFace Hub First
            if self.allow_load_from_repo:
                client = HfApi(
                    endpoint=os.environ.get("HF_ENDPOINT", None),
                    token=os.environ.get("HF_TOKEN", None),
                )
                # download the snapshot from the HuggingFace Hub
                if type_str.count("/") <= 1:
                    try:
                        assert client.repo_exists(type_str)
                        repo_info = client.repo_info(type_str)
                        assert repo_info is not None
                        repo_id = repo_info.id
                        dir_name = os.path.join(
                            FLEXRAG_CACHE_DIR,
                            f"{repo_id.split('/')[0]}--{repo_id.split('/')[1]}",
                        )
                        snapshot = client.snapshot_download(
                            repo_id=repo_id,
                            local_dir=dir_name,
                        )
                        assert snapshot is not None
                        return load_item(snapshot)
                    except AssertionError:
                        pass
                # load the item from the local repository
                elif os.path.exists(type_str):
                    # prepare the cls
                    id_path = os.path.join(type_str, "cls.id")
                    with open(id_path, "r") as f:
                        cls_name = f.read().strip()
                    # the configure will be ignored
                    # cfg_name = f"{self[cls_name]['short_names'][0]}_config"
                    # new_cfg = getattr(config, cfg_name, None)
                    # load the item
                    return self[cls_name]["item"].load_from_local(type_str)

            # Load the item directly
            if type_str in self:
                cfg_name = f"{self[type_str]['short_names'][0]}_config"
                sub_cfg = getattr(config, cfg_name, None)
                if sub_cfg is None:
                    loaded = self[type_str]["item"](**kwargs)
                else:
                    loaded = self[type_str]["item"](sub_cfg, **kwargs)
            else:
                raise ValueError(f"Invalid {self.name} type: {type_str}")
            return loaded

        choice = getattr(config, f"{self.name}_type", None)
        if choice is None:
            return None
        if isinstance(choice, (list, ListConfig)):
            loaded = []
            for name in choice:
                loaded.append(load_item(str(name)))
            return loaded
        return load_item(str(choice))

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self.names

    def __str__(self) -> str:
        data = {
            "name": self.name,
            "items": [
                {
                    "main_name": k,
                    "short_names": v["short_names"],
                    "config_class": str(v["config_class"]),
                }
                for k, v in self._items.items()
            ],
        }
        return json.dumps(data, indent=4)

    def __repr__(self) -> str:
        return str(self)

    def __add__(self, register: "Register"):
        new_register = Register()
        new_register._items = {**self._items, **register._items}
        new_register._shortcuts = {**self._shortcuts, **register._shortcuts}
        return new_register
