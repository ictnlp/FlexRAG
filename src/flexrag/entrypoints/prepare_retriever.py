from __future__ import annotations

from dataclasses import field

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.common import LOGGER_MANAGER, configure, extract_config
from flexrag.retrievers import FlexRetrieverConfig

from ._retriever_builder import (
    CollectionBackendInitConfig,
    ContextStoreInitConfig,
    build_flex_retriever,
    iter_jsonl_contexts,
)

logger = LOGGER_MANAGER.get_logger("flexrag.prepare_retriever")


@configure
class Config:
    """Configuration for building a single-backend retriever from JSONL.

    :param corpus_path: Line-delimited JSON file containing contexts.
    :param id_field: JSON row field used as ``context_id`` when absent.
    :param source_field: Optional JSON row field used as ``source``.
    :param metadata_field: Optional JSON row field used as ``meta_data``.
    :param reinit: Whether to clear existing store/backend artifacts first.
    :param context_store: Context store construction configuration.
    :param backend: Backend construction configuration.
    :param retriever_config: FlexRetriever orchestration configuration.
    """

    corpus_path: str | None = None
    id_field: str = "id"
    source_field: str | None = None
    metadata_field: str | None = None
    reinit: bool = False
    context_store: ContextStoreInitConfig = field(
        default_factory=ContextStoreInitConfig
    )
    backend: CollectionBackendInitConfig = field(
        default_factory=CollectionBackendInitConfig
    )
    retriever_config: FlexRetrieverConfig = field(default_factory=FlexRetrieverConfig)


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config) -> None:
    cfg = extract_config(cfg, Config)
    if cfg.corpus_path is None:
        raise ValueError("corpus_path must be provided.")
    retriever = build_flex_retriever(
        backend_config=cfg.backend,
        context_store_config=cfg.context_store,
        retriever_config=cfg.retriever_config,
    )
    try:
        if cfg.reinit:
            logger.warning("Reinitializing retriever artifacts.")
            retriever.clear()
        retriever.add_contexts(
            iter_jsonl_contexts(
                cfg.corpus_path,
                id_field=cfg.id_field,
                source_field=cfg.source_field,
                metadata_field=cfg.metadata_field,
            )
        )
    finally:
        for backend in retriever.backends.values():
            backend.close()
        if retriever.context_store is not None:
            retriever.context_store.close()
    return


if __name__ == "__main__":
    main()
