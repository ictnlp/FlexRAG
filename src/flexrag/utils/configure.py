import copyreg
import json
import os
from dataclasses import field, fields, is_dataclass, make_dataclass
from enum import StrEnum
from functools import partial
from typing import Generic, Iterable, Optional, Type, TypeVar

import numpy as np
from huggingface_hub import HfApi
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .default_vars import FLEXRAG_CACHE_DIR

T = TypeVar("T", bound="ConfigureBase")


class ConfigureBase:
    @classmethod
    def extract(cls: Type[T], instance: object) -> T:
        """Extract the configuration from another instance."""
        if (not is_dataclass(instance)) and (not isinstance(instance, DictConfig)):
            raise TypeError("Input must be a dataclass or DictConfig instance.")
        field_names = {f.name for f in fields(cls)}
        kwargs = {name: getattr(instance, name) for name in field_names}
        return cls(**kwargs)


def _enum_as_str(obj: StrEnum):
    """A helper function for pickle to serialize the StrEnum."""
    return (str, (obj.value,))


def Choices(choices: Iterable[str]):
    dynamic_enum = StrEnum("Choices", {c: c for c in choices})
    copyreg.pickle(dynamic_enum, _enum_as_str)
    return dynamic_enum


# Monkey Patching the JSONEncoder to handle StrEnum
class _CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, StrEnum):
            return str(obj)
        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if hasattr(obj, "to_list"):
            return obj.to_list()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


json.dumps = partial(json.dumps, cls=_CustomEncoder)
json.dump = partial(json.dump, cls=_CustomEncoder)

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
            config_fields = [(choice_name, list[str], field(default_factory=list))]
        else:
            config_fields = [(choice_name, Optional[str], field(default=default))]
        config_fields += [
            (
                f"{self[name]['short_names'][0]}_config",
                self[name]["config_class"],
                field(default_factory=self._items[name]["config_class"]),
            )
            for name in self.mainnames
            if self[name]["config_class"] is not None
        ]
        generated_config = make_dataclass(
            config_name, config_fields, bases=(ConfigureBase,)
        )

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
