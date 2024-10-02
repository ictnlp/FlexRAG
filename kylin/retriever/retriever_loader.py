from .retriever_base import (
    SEMANTIC_RETRIEVERS,
    SPARSE_RETRIEVERS,
    WEB_RETRIEVERS,
    Retriever,
)
from kylin.utils import Choices
from dataclasses import make_dataclass, field


semantic_retriever_fields = [
    (
        "retriever_type",
        Choices(SEMANTIC_RETRIEVERS.names),
        field(default=SEMANTIC_RETRIEVERS.names[0]),
    )
]
semantic_retriever_fields += [
    (
        f"{SEMANTIC_RETRIEVERS[name]['short_names'][0]}_config",
        SEMANTIC_RETRIEVERS[name]["config_class"],
        field(default_factory=SEMANTIC_RETRIEVERS[name]["config_class"]),
    )
    for name in SEMANTIC_RETRIEVERS.mainnames
]
SemanticRetrieverConfig = make_dataclass("RetrieverConfig", semantic_retriever_fields)


sparse_retriever_fields = [
    (
        "retriever_type",
        Choices(SPARSE_RETRIEVERS.names),
        field(default=SPARSE_RETRIEVERS.names[0]),
    )
]
sparse_retriever_fields += [
    (
        f"{SPARSE_RETRIEVERS[name]['short_names'][0]}_config",
        SPARSE_RETRIEVERS[name]["config_class"],
        field(default_factory=SPARSE_RETRIEVERS[name]["config_class"]),
    )
    for name in SPARSE_RETRIEVERS.mainnames
]
SparseRetrieverConfig = make_dataclass("SparseRetrieverConfig", sparse_retriever_fields)


web_retriever_fields = [
    (
        "retriever_type",
        Choices(WEB_RETRIEVERS.names),
        field(default=WEB_RETRIEVERS.names[0]),
    )
]
web_retriever_fields += [
    (
        f"{WEB_RETRIEVERS[name]['short_names'][0]}_config",
        WEB_RETRIEVERS[name]["config_class"],
        field(default_factory=WEB_RETRIEVERS[name]["config_class"]),
    )
    for name in WEB_RETRIEVERS.mainnames
]
WebRetrieverConfig = make_dataclass("WebRetrieverConfig", web_retriever_fields)


LOCAL_RETRIEVERS = SEMANTIC_RETRIEVERS + SPARSE_RETRIEVERS
local_retriever_fields = [
    (
        "retriever_type",
        Choices(LOCAL_RETRIEVERS.names),
        field(default=LOCAL_RETRIEVERS.names[0]),
    )
]
local_retriever_fields += [
    (
        f"{LOCAL_RETRIEVERS[name]['short_names'][0]}_config",
        LOCAL_RETRIEVERS[name]["config_class"],
        field(default_factory=LOCAL_RETRIEVERS[name]["config_class"]),
    )
    for name in LOCAL_RETRIEVERS.mainnames
]
LocalRetrieverConfig = make_dataclass("RetrieverConfig", local_retriever_fields)


RETRIEVERS = SEMANTIC_RETRIEVERS + SPARSE_RETRIEVERS + WEB_RETRIEVERS
retriever_fields = [
    ("retriever_type", Choices(RETRIEVERS.names), field(default=RETRIEVERS.names[0]))
]
retriever_fields += [
    (
        f"{RETRIEVERS[name]['short_names'][0]}_config",
        RETRIEVERS[name]["config_class"],
        field(default_factory=RETRIEVERS[name]["config_class"]),
    )
    for name in RETRIEVERS.mainnames
]
RetrieverConfig = make_dataclass("RetrieverConfig", retriever_fields)


def load_retriever(
    cfg: (
        SemanticRetrieverConfig  # type: ignore
        | SparseRetrieverConfig  # type: ignore
        | WebRetrieverConfig  # type: ignore
        | LocalRetrieverConfig  # type: ignore
        | RetrieverConfig  # type: ignore
    ),
) -> Retriever:
    if cfg.retriever_type in RETRIEVERS:
        cfg_name = f"{RETRIEVERS[cfg.retriever_type]['short_names'][0]}_config"
        sub_cfg = getattr(cfg, cfg_name)
        return RETRIEVERS[cfg.retriever_type]["item"](sub_cfg)
    raise ValueError(f"Unknown retriever type: {cfg.retriever_type}")
