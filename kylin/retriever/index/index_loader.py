from dataclasses import field, make_dataclass

from kylin.utils import Choices

from .index_base import DENSE_INDEX, DenseIndex


dense_index_fields = [
    (
        "index_type",
        Choices(DENSE_INDEX.names),
        field(default=DENSE_INDEX.names[0]),
    )
]
dense_index_fields += [
    (
        f"{DENSE_INDEX[name]['short_names'][0]}_config",
        DENSE_INDEX[name]["config_class"],
        field(default_factory=DENSE_INDEX[name]["config_class"]),
    )
    for name in DENSE_INDEX.mainnames
]
DenseIndexConfig = make_dataclass("DenseIndexConfig", dense_index_fields)


def load_index(index_path: str, embedding_size: int, cfg: DenseIndexConfig) -> DenseIndex:  # type: ignore
    if cfg.index_type in DENSE_INDEX:
        cfg_name = f"{DENSE_INDEX[cfg.index_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return DENSE_INDEX[cfg.index_type]["item"](index_path, embedding_size, sub_cfg)
    raise ValueError(f"Unknown index type: {cfg.index_type}")
