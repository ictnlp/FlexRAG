from dataclasses import field, make_dataclass

from kylin.utils import Choices

from .ranker import RankerBase, Rankers


ranker_fields = [
    ("ranker_type", Choices(Rankers.names), field(default=Rankers.names[0]))
]
ranker_fields += [
    (
        f"{Rankers[name]['short_names'][0]}_config",
        Rankers[name]["config_class"],
        field(default_factory=Rankers[name]["config_class"]),
    )
    for name in Rankers.mainnames
]
RankerConfig = make_dataclass("RankerConfig", ranker_fields)


def load_ranker(cfg: RankerConfig) -> RankerBase:  # type: ignore
    if cfg.ranker_type in Rankers:
        cfg_name = f"{Rankers[cfg.ranker_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Rankers[cfg.ranker_type]["item"](sub_cfg)
    raise ValueError(f"Unknown ranker type: {cfg.ranker_type}")
