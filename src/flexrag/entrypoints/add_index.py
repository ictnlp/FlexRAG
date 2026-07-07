from __future__ import annotations

from dataclasses import field

import hydra
from hydra.core.config_store import ConfigStore

from flexrag.common import configure, extract_config
from flexrag.retrievers import FlexRetriever, FlexRetrieverConfig

from ._retriever_builder import (
    CollectionBackendInitConfig,
    ContextStoreInitConfig,
    build_backend,
    build_context_store,
)


@configure
class Config:
    """Configuration for rebuilding one backend from an existing context store.

    The entrypoint keeps the historical module name but now operates on the new
    backend abstraction. It does not persist a retriever-level manifest.

    :param context_store: Existing context store used as rebuild source.
    :param backend: Backend construction configuration.
    :param retriever_config: FlexRetriever orchestration configuration.
    :param rebuild: Whether to rebuild the backend immediately.
    """

    context_store: ContextStoreInitConfig = field(
        default_factory=ContextStoreInitConfig
    )
    backend: CollectionBackendInitConfig = field(
        default_factory=CollectionBackendInitConfig
    )
    retriever_config: FlexRetrieverConfig = field(default_factory=FlexRetrieverConfig)
    rebuild: bool = True


cs = ConfigStore.instance()
cs.store(name="default", node=Config)


@hydra.main(version_base="1.3", config_path=None, config_name="default")
def main(cfg: Config) -> None:
    cfg = extract_config(cfg, Config)
    context_store = build_context_store(cfg.context_store)
    backend = build_backend(cfg.backend)
    retriever = FlexRetriever.from_backends(
        {cfg.backend.backend_name: backend},
        context_store=context_store,
        config=cfg.retriever_config,
    )
    try:
        if cfg.rebuild:
            retriever.rebuild(cfg.backend.backend_name)
    finally:
        backend.close()
        if context_store is not None:
            context_store.close()
    return


if __name__ == "__main__":
    main()
