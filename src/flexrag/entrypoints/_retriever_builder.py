from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import field
from pathlib import Path
from typing import Annotated

from flexrag.common import Choices, Context, configure
from flexrag.models import ENCODERS, EncoderConfig
from flexrag.models.encoders.encoder_base import EncoderProtocol
from flexrag.retrievers import (
    BM25SBackend,
    BM25SBackendConfig,
    CollectionBackend,
    ContextStoreProtocol,
    ElasticBackend,
    ElasticBackendConfig,
    FaissBackend,
    FaissBackendConfig,
    FlexRetriever,
    FlexRetrieverConfig,
    LanceBackend,
    LanceBackendConfig,
    LMDBContextStore,
    LMDBContextStoreConfig,
    RetrievalView,
    SQLiteContextStore,
    SQLiteContextStoreConfig,
)


@configure
class RetrievalViewInitConfig:
    """Configuration for constructing a ``RetrievalView`` in entrypoints.

    :param name: Persisted retrieval view name.
    :param fields: Context data fields projected into backend rows.
    :param merge_method: Row score aggregation method.
    """

    name: str = "text"
    fields: list[str] = field(default_factory=lambda: ["text"])
    merge_method: Annotated[str, Choices("max", "mean", "sum", "concat")] = "max"


@configure
class ContextStoreInitConfig:
    """Configuration for constructing an entrypoint context store.

    :param context_store_type: Store implementation to construct, or ``"none"``.
    :param path: Store artifact path. Required unless type is ``"none"``.
    :param serializer: Serializer used by local stores.
    :param table_name: SQLite table name when using SQLite.
    """

    context_store_type: Annotated[str, Choices("lmdb", "sqlite", "none")] = "lmdb"
    path: str | None = None
    serializer: str = "msgpack"
    table_name: str = "contexts"


@configure
class CollectionBackendInitConfig:
    """Configuration for constructing one collection backend in entrypoints.

    :param backend_name: Retriever-owned backend name.
    :param backend_type: Backend implementation to construct.
    :param path: Local backend artifact path for BM25S or Faiss.
    :param lance_uri: LanceDB URI when using Lance.
    :param view_config: Retrieval view bound to the backend.
    :param query_encoder_config: Encoder used by dense backends.
    :param passage_encoder_config: Optional separate passage encoder.
    """

    backend_name: str = "default"
    backend_type: Annotated[
        str,
        Choices("bm25s", "faiss", "elastic", "lance"),
    ] = "bm25s"
    path: str | None = None
    lance_uri: str | None = None
    view_config: RetrievalViewInitConfig = field(
        default_factory=RetrievalViewInitConfig
    )
    bm25s_config: BM25SBackendConfig = field(default_factory=BM25SBackendConfig)
    faiss_config: FaissBackendConfig = field(default_factory=FaissBackendConfig)
    elastic_config: ElasticBackendConfig = field(
        default_factory=lambda: ElasticBackendConfig(index_name="contexts")
    )
    lance_config: LanceBackendConfig = field(default_factory=LanceBackendConfig)
    query_encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    passage_encoder_config: EncoderConfig | None = None


def build_retrieval_view(config: RetrievalViewInitConfig) -> RetrievalView:
    """Build a retrieval view from entrypoint configuration.

    :param config: View construction configuration.
    :return: Retrieval view instance.
    """
    return RetrievalView(
        name=config.name,
        fields=list(config.fields),
        merge_method=config.merge_method,
    )


def build_context_store(
    config: ContextStoreInitConfig,
) -> ContextStoreProtocol | None:
    """Build a context store from entrypoint configuration.

    :param config: Context store construction configuration.
    :return: Context store instance, or ``None``.
    :raises ValueError: If the selected store requires a path and none is set.
    """
    if config.context_store_type == "none":
        return None
    if config.path is None:
        raise ValueError("context_store.path must be provided.")
    if config.context_store_type == "lmdb":
        return LMDBContextStore(
            LMDBContextStoreConfig(
                path=config.path,
                serializer=config.serializer,
            )
        )
    if config.context_store_type == "sqlite":
        return SQLiteContextStore(
            SQLiteContextStoreConfig(
                path=config.path,
                serializer=config.serializer,
                table_name=config.table_name,
            )
        )
    raise ValueError(f"Unknown context_store_type: {config.context_store_type}")


def build_backend(config: CollectionBackendInitConfig) -> CollectionBackend:
    """Build one collection backend from entrypoint configuration.

    :param config: Backend construction configuration.
    :return: Collection backend instance.
    :raises ValueError: If required local paths or dense encoders are missing.
    """
    view = build_retrieval_view(config.view_config)
    if config.backend_type == "bm25s":
        if config.path is None:
            raise ValueError("backend.path must be provided for BM25SBackend.")
        return BM25SBackend(view, config.path, config.bm25s_config)
    if config.backend_type == "faiss":
        if config.path is None:
            raise ValueError("backend.path must be provided for FaissBackend.")
        query_encoder, passage_encoder = _build_dense_encoders(config)
        return FaissBackend(
            view,
            config.path,
            query_encoder=query_encoder,
            passage_encoder=passage_encoder,
            config=config.faiss_config,
        )
    if config.backend_type == "elastic":
        query_encoder = None
        passage_encoder = None
        if config.elastic_config.retrieval_mode == "dense":
            query_encoder, passage_encoder = _build_dense_encoders(config)
        return ElasticBackend(
            view,
            config.elastic_config,
            query_encoder=query_encoder,
            passage_encoder=passage_encoder,
        )
    if config.backend_type == "lance":
        if config.lance_uri is None:
            raise ValueError("backend.lance_uri must be provided for LanceBackend.")
        query_encoder = None
        passage_encoder = None
        if config.lance_config.retrieval_mode == "dense":
            query_encoder, passage_encoder = _build_dense_encoders(config)
        return LanceBackend(
            view,
            config.lance_uri,
            config.lance_config,
            query_encoder=query_encoder,
            passage_encoder=passage_encoder,
        )
    raise ValueError(f"Unknown backend_type: {config.backend_type}")


def build_flex_retriever(
    *,
    backend_config: CollectionBackendInitConfig,
    context_store_config: ContextStoreInitConfig,
    retriever_config: FlexRetrieverConfig | None = None,
) -> FlexRetriever:
    """Build a single-backend ``FlexRetriever`` for legacy entrypoints.

    :param backend_config: Backend construction configuration.
    :param context_store_config: Context store construction configuration.
    :param retriever_config: Optional retriever orchestration configuration.
    :return: FlexRetriever instance.
    """
    context_store = build_context_store(context_store_config)
    backend = build_backend(backend_config)
    return FlexRetriever.from_backends(
        {backend_config.backend_name: backend},
        context_store=context_store,
        config=retriever_config,
    )


def iter_jsonl_contexts(
    path: str | Path,
    *,
    id_field: str = "id",
    source_field: str | None = None,
    metadata_field: str | None = None,
) -> Iterable[Context]:
    """Yield contexts from a line-delimited JSON file.

    Rows that already contain a ``data`` mapping are interpreted as serialized
    contexts. Otherwise, all non-control fields become ``Context.data``.

    :param path: JSONL file path.
    :param id_field: Row field used as ``context_id`` when ``context_id`` is
        absent.
    :param source_field: Optional row field used as ``source``.
    :param metadata_field: Optional row field used as ``meta_data``.
    :return: Context iterator.
    """
    control_fields = {
        "context_id",
        id_field,
        "data",
        "source",
        "meta_data",
    }
    if source_field is not None:
        control_fields.add(source_field)
    if metadata_field is not None:
        control_fields.add(metadata_field)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Each JSONL row must be an object.")
            data = row.get("data")
            if not isinstance(data, dict):
                data = {
                    key: value
                    for key, value in row.items()
                    if key not in control_fields
                }
            source = row.get(source_field) if source_field is not None else row.get("source")
            meta_data = (
                row.get(metadata_field)
                if metadata_field is not None
                else row.get("meta_data", {})
            )
            if not isinstance(meta_data, dict):
                meta_data = {"value": meta_data}
            yield Context(
                context_id=row.get("context_id", row.get(id_field)),
                data=data,
                source=source,
                meta_data=meta_data,
            )


def _build_dense_encoders(
    config: CollectionBackendInitConfig,
) -> tuple[EncoderProtocol, EncoderProtocol | None]:
    query_encoder = ENCODERS.load(config.query_encoder_config)
    if query_encoder is None:
        raise ValueError("query_encoder_config must be configured for dense backends.")
    passage_encoder = None
    if config.passage_encoder_config is not None:
        passage_encoder = ENCODERS.load(config.passage_encoder_config)
    return query_encoder, passage_encoder
