from dataclasses import field, make_dataclass

from kylin.utils import Choices

from .searcher import Searchers, BaseSearcher


searcher_fields = [
    ("searcher_type", Choices(Searchers.names), field(default=Searchers.names[0]))
]
searcher_fields += [
    (
        f"{Searchers[name]['short_names'][0]}_searcher_config",
        Searchers[name]["config_class"],
        field(default_factory=Searchers[name]["config_class"]),
    )
    for name in Searchers.mainnames
]
SearcherConfig = make_dataclass("SearcherConfig", searcher_fields)


def load_searcher(cfg: SearcherConfig) -> BaseSearcher:  # type: ignore
    if cfg.searcher_type in Searchers:
        cfg_name = f"{Searchers[cfg.searcher_type]['short_names'][0]}_searcher_config"
        sub_cfg = getattr(cfg, cfg_name)
        return Searchers[cfg.searcher_type]["item"](sub_cfg)
    raise ValueError(f"Unknown searcher type: {cfg.searcher_type}")
