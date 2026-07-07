from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Iterable

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.models.encoders.encoder_base import EncoderProtocol

from ..context_store.payload import context_to_payload, payload_to_context
from ..utils import _iter_batches
from ..view import RetrievalViewConfig
from .base import AsyncCollectionBackendBase, Hit

INTERNAL_CONTEXT_ID = "_context_id"
INTERNAL_VIEW = "_view"
INTERNAL_FIELD = "_field"
INTERNAL_TEXT = "_text"
INTERNAL_VECTOR = "_vector"
INTERNAL_PAYLOAD = "_payload"
INTERNAL_BACKEND_METADATA = "flexrag.backend"
INTERNAL_METADATA_VERSION = 1


@configure
class LanceBackendConfig:
    """Configuration for ``LanceBackend``.

    :param uri: LanceDB connection URI. Required at construction time.
    :param view: Retrieval view configuration. Required for new tables; may be
        omitted when loading a table that already persists its view.
    :param table_name: LanceDB table name.
    :param retrieval_mode: Single retrieval mode, ``"sparse"`` or ``"dense"``.
    :param store_payload: Whether rows store native context payloads.
    :param batch_size: Rebuild batch size.
    :param vector_metric: Dense vector distance metric.
    :param vector_index_type: LanceDB vector index config class name.
    :param index_options: Extra index construction options persisted in schema.
    :param search_options: Default search-time options, not persisted.
    :param connect_options: Runtime-only options passed to LanceDB connect.
    """

    uri: str | Path | None = None
    view: RetrievalViewConfig | None = None
    table_name: str = "contexts"
    retrieval_mode: Annotated[str, Choices("sparse", "dense")] = "dense"
    store_payload: bool = True
    batch_size: int = 1000
    vector_metric: Annotated[str, Choices("l2", "cosine", "dot")] = "cosine"
    vector_index_type: str = "IvfFlat"
    index_options: dict[str, Any] | None = None
    search_options: dict[str, Any] | None = None
    connect_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.retrieval_mode not in {"sparse", "dense"}:
            raise ValueError(f"Invalid retrieval_mode: {self.retrieval_mode}")
        if self.vector_metric not in {"l2", "cosine", "dot"}:
            raise ValueError(f"Invalid vector_metric: {self.vector_metric}")
        return


class LanceBackend(AsyncCollectionBackendBase):
    """Async-native LanceDB backend for sparse or dense retrieval.

    The backend stores one projected row per retrieval field. Dense mode uses
    runtime encoders; hybrid retrieval is handled by ``FlexRetriever``.
    """

    requires_context_store = False
    is_addable = True

    def __init__(
        self,
        config: LanceBackendConfig,
        *,
        client: Any | None = None,
        query_encoder: EncoderProtocol | None = None,
        passage_encoder: EncoderProtocol | None = None,
    ) -> None:
        """Create or attach to a LanceDB table.

        :param config: Lance backend configuration.
        :param client: Optional async LanceDB-compatible connection.
        :param query_encoder: Runtime encoder required for dense mode queries.
        :param passage_encoder: Optional encoder for dense projected content.
        :raises ValueError: If dense mode lacks an encoder or persisted metadata
            conflicts with the provided view/config.
        """
        if config.uri is None:
            raise ValueError("LanceBackendConfig.uri must be provided.")
        super().__init__(config.view.to_view() if config.view is not None else None)
        self.uri = os.fspath(config.uri)
        self.config = config
        if self.config.retrieval_mode == "dense" and query_encoder is None:
            raise ValueError("LanceBackend dense mode requires query_encoder.")
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.client = client
        self.vector_dims: int | None = None
        self._run_coroutine_sync(self._async_init())
        self._require_view()
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Add contexts by appending projected Lance rows.

        Sparse mode requires text projection content. Dense mode encodes
        projected content with ``passage_encoder``.

        :param contexts: Context objects to append.
        :raises TypeError: If sparse mode receives non-text content.
        """
        items = list(contexts)
        if not items:
            return
        rows, vector_dims = self._contexts_to_rows(items)
        if not rows:
            return
        table = await self._get_table()
        if table is None:
            if self.config.retrieval_mode == "dense":
                self.vector_dims = vector_dims
            table = await self.client.create_table(self.config.table_name, data=rows)
        else:
            await self._async_load_persisted_view_if_present()
            self._validate_vector_dims(vector_dims)
            await table.add(rows)
        await self._async_write_metadata()
        await self._async_create_index()
        return

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Drop and rebuild the Lance table from the provided corpus.

        :param contexts: Complete context corpus to project and index.
        """
        batches = _iter_batches(contexts, self.config.batch_size)
        await self.async_clear()
        self.vector_dims = None
        for batch in batches:
            await self.async_add_contexts(batch)
        return

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Search LanceDB and return lightweight hits.

        :param queries: Text queries in sparse mode, encoder inputs in dense mode.
        :param top_k: Maximum hits per query.
        :param search_options: Optional Lance search-time overrides.
        :returns: One hit list per query.
        """
        if top_k <= 0:
            return [[] for _ in queries]
        table = await self._get_table()
        if table is None:
            return [[] for _ in queries]
        await self._async_load_persisted_view_if_present()
        if self.config.retrieval_mode == "dense":
            return await self._async_search_dense_hits(
                table,
                queries,
                top_k,
                search_options=search_options,
            )
        return await self._async_search_sparse_hits(
            table,
            queries,
            top_k,
            search_options=search_options,
        )

    async def async_clear(self) -> None:
        """Drop the Lance table for this backend if it exists."""
        if await self._table_exists():
            await self.client.drop_table(
                self.config.table_name,
                ignore_missing=True,
            )
        self.vector_dims = None
        return

    async def async_get_context(self, context_id: str) -> Context:
        """Hydrate a context from native Lance payloads.

        :param context_id: Context identifier to fetch.
        :returns: Stored context payload.
        :raises KeyError: If payload storage is disabled or the id is missing.
        """
        table = await self._get_table()
        if table is None:
            raise KeyError(context_id)
        builder = table.query()
        rows = await builder.where(
            f"{INTERNAL_CONTEXT_ID} = {self._sql_string(context_id)}"
        ).limit(1).to_list()
        if not rows:
            raise KeyError(context_id)
        payload = rows[0].get(INTERNAL_PAYLOAD)
        if payload is None:
            raise KeyError(context_id)
        return payload_to_context(json.loads(payload))

    async def async_count(self) -> int:
        """Count unique context ids for the current persisted view.

        :returns: Unique ``context_id`` count.
        """
        table = await self._get_table()
        if table is None:
            return 0
        row_count = await table.count_rows()
        if row_count == 0:
            return 0
        rows = await table.query().where(self._view_where(None)).limit(row_count).to_list()
        return len({row[INTERNAL_CONTEXT_ID] for row in rows})

    async def async_close(self) -> None:
        """Close the underlying LanceDB client connection."""
        self.client.close()
        return

    async def _async_init(self) -> None:
        if self.client is None:
            try:
                import lancedb
            except ImportError as exc:
                raise ImportError(
                    "Please install LanceDB before using LanceBackend."
                ) from exc
            self.client = await lancedb.connect_async(
                self.uri,
                **(self.config.connect_options or {}),
            )
        await self._async_load_persisted_view_if_present()
        return

    async def _async_search_sparse_hits(
        self,
        table: Any,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None,
    ) -> list[list[Hit]]:
        if any(not isinstance(query, str) for query in queries):
            raise TypeError("LanceBackend sparse mode only supports text queries.")
        row_k = max(top_k, top_k * max(1, len(self._require_view().fields)))
        results = []
        for query in queries:
            builder = await table.search(
                query,
                query_type="fts",
                fts_columns=INTERNAL_TEXT,
            )
            builder = self._apply_search_options(
                builder,
                search_options,
                dense=False,
            )
            rows = await builder.limit(row_k).to_list()
            results.append(self._merge_rows(rows, top_k))
        return results

    async def _async_search_dense_hits(
        self,
        table: Any,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None,
    ) -> list[list[Hit]]:
        view = self._require_view()
        query_vectors = self.query_encoder.encode(queries)
        row_k = max(top_k, top_k * max(1, len(view.fields)))
        results = []
        for vector in query_vectors:
            vector_value = vector.tolist()
            if self.vector_dims is not None and len(vector_value) != self.vector_dims:
                raise ValueError(
                    "Encoded query vector size does not match the Lance table schema."
                )
            builder = await table.search(
                vector_value,
                vector_column_name=INTERNAL_VECTOR,
                query_type="vector",
            )
            builder = builder.distance_type(self.config.vector_metric)
            builder = self._apply_search_options(
                builder,
                search_options,
                dense=True,
            )
            rows = await builder.limit(row_k).to_list()
            results.append(self._merge_rows(rows, top_k))
        return results

    def _contexts_to_rows(
        self,
        contexts: list[Context],
    ) -> tuple[list[dict[str, Any]], int | None]:
        view = self._require_view()
        if self.config.retrieval_mode == "dense":
            projected = [
                (context, row)
                for context in contexts
                for row in view.project(context)
            ]
            if not projected:
                return [], None
            embeddings = self.passage_encoder.encode(
                [row.content for _, row in projected]
            )
            rows = [
                self._row_dict(
                    context,
                    row.context_id,
                    row.field,
                    vector=embedding.tolist(),
                )
                for (context, row), embedding in zip(projected, embeddings)
            ]
            return rows, embeddings.shape[1]

        rows = []
        for context in contexts:
            for row in view.project(context):
                if not isinstance(row.content, str):
                    raise TypeError(
                        "LanceBackend sparse mode only supports text retrieval "
                        f"content; field {row.field!r} has type "
                        f"{type(row.content).__name__}."
                    )
                rows.append(
                    self._row_dict(
                        context,
                        row.context_id,
                        row.field,
                        text=row.content,
                    )
                )
        return rows, None

    def _row_dict(
        self,
        context: Context,
        context_id: str,
        field: str,
        *,
        text: str | None = None,
        vector: list[float] | None = None,
    ) -> dict[str, Any]:
        row = {
            INTERNAL_CONTEXT_ID: context_id,
            INTERNAL_VIEW: self._require_view().name,
            INTERNAL_FIELD: field,
        }
        if text is not None:
            row[INTERNAL_TEXT] = text
        if vector is not None:
            row[INTERNAL_VECTOR] = vector
        if self.config.store_payload:
            row[INTERNAL_PAYLOAD] = json.dumps(
                context_to_payload(context),
                ensure_ascii=False,
            )
        return row

    async def _async_load_persisted_view_if_present(self) -> bool:
        table = await self._get_table()
        if table is None:
            return False
        schema = await table.schema()
        raw = self._metadata_value_from_schema(schema, self._content_column())
        if raw is None:
            for column in (INTERNAL_TEXT, INTERNAL_VECTOR):
                if column == self._content_column():
                    continue
                raw = self._metadata_value_from_schema(schema, column)
                if raw is not None:
                    break
        if raw is None:
            return False
        payload = json.loads(raw)
        if payload["version"] != INTERNAL_METADATA_VERSION:
            raise ValueError(
                "Unsupported LanceBackend metadata version: "
                f"{payload['version']!r}"
            )
        self._load_persisted_view(payload["view"])
        self._load_persisted_schema(payload)
        return True

    async def _async_write_metadata(self) -> None:
        table = await self._get_table()
        if table is None:
            return
        schema = await table.schema()
        metadata = self._field_metadata(schema, self._content_column())
        metadata[INTERNAL_BACKEND_METADATA] = json.dumps(
            self._metadata_dict(),
            ensure_ascii=False,
            sort_keys=True,
        )
        await table.update_field_metadata(
            {
                "path": self._content_column(),
                "metadata": metadata,
                "replace": True,
            }
        )
        return

    async def _async_create_index(self) -> None:
        table = await self._get_table()
        if table is None or await table.count_rows() == 0:
            return
        try:
            await table.create_index(
                self._content_column(),
                config=self._index_config(),
                replace=True,
                name=self._physical_index_name(),
            )
        except Exception as exc:
            if self.config.retrieval_mode == "dense":
                warnings.warn(
                    "LanceDB vector index creation failed; exact vector search can "
                    f"still work. Error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return
            raise
        return

    async def _table_exists(self) -> bool:
        return self.config.table_name in await self.client.table_names()

    async def _get_table(self) -> Any | None:
        if not await self._table_exists():
            return None
        return await self.client.open_table(self.config.table_name)

    def _metadata_dict(self) -> dict[str, Any]:
        return {
            "version": INTERNAL_METADATA_VERSION,
            "view": self._require_view().to_dict(),
            "retrieval_mode": self.config.retrieval_mode,
            "store_payload": self.config.store_payload,
            "vector_dims": self.vector_dims,
            "vector_metric": (
                self.config.vector_metric
                if self.config.retrieval_mode == "dense"
                else None
            ),
            "vector_index_type": (
                self.config.vector_index_type
                if self.config.retrieval_mode == "dense"
                else None
            ),
            "index_options": self.config.index_options,
        }

    def _load_persisted_schema(self, payload: dict[str, Any]) -> None:
        if payload["retrieval_mode"] != self.config.retrieval_mode:
            raise ValueError(
                "LanceBackend retrieval_mode does not match persisted schema."
            )
        if payload["store_payload"] != self.config.store_payload:
            raise ValueError(
                "LanceBackend store_payload does not match persisted schema."
            )
        if self.config.retrieval_mode == "sparse":
            return
        if payload["vector_metric"] != self.config.vector_metric:
            raise ValueError(
                "LanceBackend vector_metric does not match persisted schema."
            )
        if payload["vector_index_type"] != self.config.vector_index_type:
            raise ValueError(
                "LanceBackend vector_index_type does not match persisted schema."
            )
        if payload["index_options"] != self.config.index_options:
            raise ValueError(
                "LanceBackend index_options do not match persisted schema."
            )
        vector_dims = payload["vector_dims"]
        if vector_dims is not None:
            self.vector_dims = int(vector_dims)
        return

    def _validate_vector_dims(self, vector_dims: int | None) -> None:
        if self.config.retrieval_mode != "dense" or vector_dims is None:
            return
        if self.vector_dims is None:
            self.vector_dims = vector_dims
            return
        if self.vector_dims != vector_dims:
            raise ValueError(
                "Encoded vector size does not match the persisted Lance schema."
            )
        return

    def _index_config(self) -> Any:
        if self.config.retrieval_mode == "sparse":
            from lancedb.index import FTS

            return FTS(**(self.config.index_options or {}))
        import lancedb.index as lancedb_index

        config_cls = getattr(lancedb_index, self.config.vector_index_type, None)
        if config_cls is None:
            raise ValueError(
                f"Unsupported Lance vector_index_type: {self.config.vector_index_type}"
            )
        return config_cls(
            distance_type=self.config.vector_metric,
            **(self.config.index_options or {}),
        )

    def _apply_search_options(
        self,
        builder: Any,
        search_options: dict[str, Any] | None,
        *,
        dense: bool,
    ) -> Any:
        options = dict(self.config.search_options or {})
        options.update(search_options or {})
        forbidden = {"limit", "query", "vector_column", "vector_column_name"}
        conflicts = forbidden.intersection(options)
        if conflicts:
            raise ValueError(
                "LanceBackend search_options cannot override "
                f"{sorted(conflicts)}."
            )
        where = options.pop("where", None)
        fast_search = options.pop("fast_search", False)
        if dense:
            nprobes = options.pop("nprobes", None)
            refine_factor = options.pop("refine_factor", None)
            if nprobes is not None:
                builder = builder.nprobes(nprobes)
            if refine_factor is not None:
                builder = builder.refine_factor(refine_factor)
        builder = builder.where(self._view_where(where))
        if fast_search:
            builder = builder.fast_search()
        if options:
            raise ValueError(
                "Unsupported LanceBackend search_options: "
                f"{sorted(options)}."
            )
        return builder

    def _merge_rows(self, rows: list[dict[str, Any]], top_k: int) -> list[Hit]:
        view = self._require_view()
        by_context: dict[str, list[float]] = defaultdict(list)
        contexts: dict[str, Context] = {}
        for row in rows:
            context_id = row[INTERNAL_CONTEXT_ID]
            by_context[context_id].append(self._row_score(row))
            payload = row.get(INTERNAL_PAYLOAD)
            if payload is not None:
                contexts.setdefault(context_id, payload_to_context(json.loads(payload)))
        ordered = sorted(
            (
                (context_id, view.aggregate_scores(scores))
                for context_id, scores in by_context.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            Hit(
                context_id=context_id,
                score=score,
                backend="",
                view=view.name,
                context=contexts.get(context_id),
            )
            for context_id, score in ordered[:top_k]
        ]

    @staticmethod
    def _row_score(row: dict[str, Any]) -> float:
        if row.get("_score") is not None:
            return float(row["_score"])
        if row.get("_distance") is not None:
            return -float(row["_distance"])
        return 0.0

    def _content_column(self) -> str:
        if self.config.retrieval_mode == "sparse":
            return INTERNAL_TEXT
        return INTERNAL_VECTOR

    def _physical_index_name(self) -> str:
        prefix = "fts" if self.config.retrieval_mode == "sparse" else "vec"
        return f"{prefix}__{self._require_view().name}"

    def _view_where(self, where: str | None) -> str:
        view_filter = f"{INTERNAL_VIEW} = {self._sql_string(self._require_view().name)}"
        if where is None:
            return view_filter
        return f"({view_filter}) AND ({where})"

    @staticmethod
    def _sql_string(value: str) -> str:
        return "'" + value.replace("'", "''") + "'"

    @staticmethod
    def _metadata_value_from_schema(schema: Any, field_name: str) -> str | None:
        try:
            metadata = LanceBackend._field_metadata(schema, field_name)
        except KeyError:
            return None
        return metadata.get(INTERNAL_BACKEND_METADATA)

    @staticmethod
    def _field_metadata(schema: Any, field_name: str) -> dict[str, str]:
        field = schema.field(field_name)
        if not field.metadata:
            return {}
        return {
            LanceBackend._metadata_value_to_str(key): LanceBackend._metadata_value_to_str(
                value
            )
            for key, value in field.metadata.items()
        }

    @staticmethod
    def _metadata_value_to_str(value: bytes | str) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value
