from __future__ import annotations

from collections import defaultdict
from typing import Annotated, Any, Iterable

from elasticsearch import AsyncElasticsearch

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import Context
from flexrag.models.encoders.encoder_base import EncoderProtocol

from ..context_store.payload import context_to_payload, payload_to_context
from ..utils import _iter_batches
from ..view import RetrievalView
from .base import AsyncCollectionBackendBase, Hit

INTERNAL_CONTEXT_ID = "_context_id"
INTERNAL_VIEW = "_view"
INTERNAL_FIELD = "_field"
INTERNAL_TEXT = "_text"
INTERNAL_VECTOR = "_vector"
INTERNAL_PAYLOAD = "_payload"
INTERNAL_VIEW_DEFINITION = "_retrieval_view"
INTERNAL_BACKEND_SCHEMA = "_backend_schema"
INTERNAL_META_CONTEXT_ID = "__flexrag_backend_meta__"
INTERNAL_META_DOC_ID = "__flexrag_backend_meta__"


@configure
class ElasticBackendConfig:
    """Configuration for ``ElasticBackend``.

    :param index_name: Elasticsearch index name.
    :param host: Elasticsearch HTTP endpoint used when no client is supplied.
    :param api_key: Optional runtime API key. It is not persisted in metadata.
    :param store_payload: Whether projection rows also store native payloads.
    :param batch_size: Bulk indexing and rebuild batch size.
    :param number_of_shards: Index shard count for new indices.
    :param number_of_replicas: Replica count for new indices.
    :param retrieval_mode: Single retrieval mode, ``"sparse"`` or ``"dense"``.
    :param vector_similarity: Dense vector similarity for new dense indices.
    :param index_options: Extra dense vector mapping options persisted in schema.
    :param search_options: Default search-time options, not persisted.
    """

    index_name: str
    host: str = "http://localhost:9200"
    api_key: str | None = None
    store_payload: bool = True
    batch_size: int = 1000
    number_of_shards: int = 1
    number_of_replicas: int = 0
    retrieval_mode: Annotated[str, Choices("sparse", "dense")] = "sparse"
    vector_similarity: Annotated[
        str,
        Choices("cosine", "dot_product", "l2_norm", "max_inner_product"),
    ] = "cosine"
    index_options: dict[str, Any] | None = None
    search_options: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.retrieval_mode not in {"sparse", "dense"}:
            raise ValueError(f"Invalid retrieval_mode: {self.retrieval_mode}")
        if self.vector_similarity not in {
            "cosine",
            "dot_product",
            "l2_norm",
            "max_inner_product",
        }:
            raise ValueError(f"Invalid vector_similarity: {self.vector_similarity}")
        return


class ElasticBackend(AsyncCollectionBackendBase):
    """Async-native Elasticsearch backend for sparse or dense retrieval.

    Each backend instance owns one retrieval mode and one persisted view. Dense
    mode uses runtime encoders and Elasticsearch dense-vector search; hybrid
    retrieval is handled by ``FlexRetriever`` multi-backend merge.
    """

    requires_context_store = False
    is_addable = True

    def __init__(
        self,
        view: RetrievalView | None,
        config: ElasticBackendConfig,
        *,
        client: Any | None = None,
        query_encoder: EncoderProtocol | None = None,
        passage_encoder: EncoderProtocol | None = None,
    ) -> None:
        """Create or attach to an Elasticsearch index.

        :param view: Retrieval view, or ``None`` to load it from index metadata.
        :param config: Elasticsearch backend configuration.
        :param client: Optional async Elasticsearch-compatible client.
        :param query_encoder: Runtime encoder required for dense mode queries.
        :param passage_encoder: Optional encoder for dense projected content.
        :raises ValueError: If dense mode lacks an encoder or persisted metadata
            conflicts with the provided view/config.
        """
        super().__init__(view)
        self.config = config
        if self.config.retrieval_mode == "dense" and query_encoder is None:
            raise ValueError("ElasticBackend dense mode requires query_encoder.")
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder or query_encoder
        self.client = client or AsyncElasticsearch(
            config.host,
            api_key=config.api_key,
        )
        self.vector_dims: int | None = None
        self._run_coroutine_sync(self._async_load_persisted_view_if_present())
        self._require_view()
        return

    async def async_add_contexts(self, contexts: Iterable[Context]) -> None:
        """Add contexts by indexing projected Elasticsearch rows.

        Sparse mode requires text projection content. Dense mode encodes
        projected content with ``passage_encoder``.

        :param contexts: Context objects to append.
        :raises TypeError: If sparse mode receives non-text content.
        """
        items = list(contexts)
        if not items:
            return
        if self.config.retrieval_mode == "dense":
            await self._async_add_dense_contexts(items)
            return
        await self._async_add_sparse_contexts(items)
        return

    async def async_rebuild(self, contexts: Iterable[Context]) -> None:
        """Clear the index and rebuild it from the provided corpus.

        :param contexts: Complete context corpus to project and index.
        """
        batches = _iter_batches(contexts, self.config.batch_size)
        await self.async_clear()
        for batch in batches:
            await self.async_add_contexts(batch)
        return

    async def _async_add_sparse_contexts(self, contexts: list[Context]) -> None:
        view = self._require_view()
        await self._ensure_index()
        operations = []
        for context in contexts:
            for row in view.project(context):
                if not isinstance(row.content, str):
                    raise TypeError(
                        "ElasticBackend only supports text retrieval content; "
                        f"field {row.field!r} has type {type(row.content).__name__}."
                    )
                doc = {
                    INTERNAL_CONTEXT_ID: row.context_id,
                    INTERNAL_VIEW: view.name,
                    INTERNAL_FIELD: row.field,
                    INTERNAL_TEXT: row.content,
                }
                if self.config.store_payload:
                    doc[INTERNAL_PAYLOAD] = context_to_payload(context)
                operations.append(
                    {
                        "index": {
                            "_index": self.config.index_name,
                            "_id": self._row_id(row.context_id, row.field),
                        }
                    }
                )
                operations.append(doc)
                if len(operations) >= self.config.batch_size * 2:
                    await self._bulk(operations)
                    operations = []
        if operations:
            await self._bulk(operations)
        return

    async def _async_add_dense_contexts(self, contexts: list[Context]) -> None:
        view = self._require_view()
        rows = []
        for context in contexts:
            for row in view.project(context):
                rows.append((context, row))
        if not rows:
            return

        assert self.passage_encoder is not None
        embeddings = self.passage_encoder.encode([row.content for _, row in rows])
        await self._ensure_index(vector_dims=embeddings.shape[1])
        operations = []
        for (context, row), embedding in zip(rows, embeddings):
            doc = {
                INTERNAL_CONTEXT_ID: row.context_id,
                INTERNAL_VIEW: view.name,
                INTERNAL_FIELD: row.field,
                INTERNAL_VECTOR: embedding.tolist(),
            }
            if self.config.store_payload:
                doc[INTERNAL_PAYLOAD] = context_to_payload(context)
            operations.append(
                {
                    "index": {
                        "_index": self.config.index_name,
                        "_id": self._row_id(row.context_id, row.field),
                    }
                }
            )
            operations.append(doc)
            if len(operations) >= self.config.batch_size * 2:
                await self._bulk(operations)
                operations = []
        if operations:
            await self._bulk(operations)
        return

    async def async_search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        """Search Elasticsearch and return lightweight hits.

        :param queries: Text queries in sparse mode, encoder inputs in dense mode.
        :param top_k: Maximum hits per query.
        :param search_options: Optional search-time overrides.
        :returns: One hit list per query.
        """
        if top_k <= 0:
            return [[] for _ in queries]
        if not await self._index_exists():
            return [[] for _ in queries]
        if self.config.retrieval_mode == "dense" and self.vector_dims is None:
            await self._async_load_persisted_view_if_present()
        if self.config.retrieval_mode == "dense":
            return await self._async_search_dense_hits(
                queries,
                top_k,
                search_options=search_options,
            )
        return await self._async_search_sparse_hits(
            queries,
            top_k,
            search_options=search_options,
        )

    async def _async_search_sparse_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        if any(not isinstance(query, str) for query in queries):
            raise TypeError("ElasticBackend only supports text queries.")
        view = self._require_view()
        body = []
        row_k = max(top_k, top_k * max(1, len(view.fields)))
        for query in queries:
            body.append({"index": self.config.index_name})
            body.append(self._build_sparse_search_body(query, row_k, search_options))
        response = await self.client.msearch(body=body)
        responses = self._body(response).get("responses", [])
        return [self._merge_response_hits(response, top_k) for response in responses]

    async def _async_search_dense_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        view = self._require_view()
        assert self.query_encoder is not None
        query_vectors = self.query_encoder.encode(queries)
        body = []
        row_k = max(top_k, top_k * max(1, len(view.fields)))
        for vector in query_vectors:
            vector_value = vector.tolist()
            if self.vector_dims is not None and len(vector_value) != self.vector_dims:
                raise ValueError(
                    "Encoded query vector size does not match the Elastic index."
                )
            body.append({"index": self.config.index_name})
            body.append(
                self._build_dense_search_body(
                    vector_value,
                    row_k,
                    search_options=search_options,
                )
            )
        response = await self.client.msearch(body=body)
        responses = self._body(response).get("responses", [])
        return [self._merge_response_hits(response, top_k) for response in responses]

    async def async_clear(self) -> None:
        """Delete the Elasticsearch index if it exists."""
        if await self._index_exists():
            await self.client.indices.delete(index=self.config.index_name)
        return

    async def async_get_context(self, context_id: str) -> Context:
        """Hydrate a context from native Elasticsearch payloads.

        :param context_id: Context identifier to fetch.
        :returns: Stored context payload.
        :raises KeyError: If payload storage is disabled or the id is missing.
        """
        if context_id == INTERNAL_META_CONTEXT_ID:
            raise KeyError(context_id)
        response = await self.client.search(
            index=self.config.index_name,
            body={
                "query": {"term": {INTERNAL_CONTEXT_ID: context_id}},
                "size": 1,
            },
        )
        hits = self._body(response).get("hits", {}).get("hits", [])
        if not hits:
            raise KeyError(context_id)
        payload = hits[0].get("_source", {}).get(INTERNAL_PAYLOAD)
        if payload is None:
            raise KeyError(context_id)
        return payload_to_context(payload)

    async def async_count(self) -> int:
        """Count unique context ids for the current persisted view.

        Metadata documents are excluded from the count.

        :returns: Unique ``context_id`` count.
        """
        if not await self._index_exists():
            return 0
        view = self._require_view()
        total = 0
        after_key = None
        while True:
            composite: dict[str, Any] = {
                "size": self.config.batch_size,
                "sources": [
                    {"context_id": {"terms": {"field": INTERNAL_CONTEXT_ID}}},
                ],
            }
            if after_key is not None:
                composite["after"] = after_key
            response = await self.client.search(
                index=self.config.index_name,
                body={
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": [{"term": {INTERNAL_VIEW: view.name}}],
                            "must_not": [
                                {"term": {INTERNAL_CONTEXT_ID: INTERNAL_META_CONTEXT_ID}}
                            ],
                        }
                    },
                    "aggs": {"contexts": {"composite": composite}},
                },
            )
            contexts = (
                self._body(response)
                .get("aggregations", {})
                .get("contexts", {})
            )
            buckets = contexts.get("buckets", [])
            total += len(buckets)
            after_key = contexts.get("after_key")
            if not buckets or after_key is None:
                return total

    async def async_close(self) -> None:
        """Close the underlying async Elasticsearch client."""
        await self.client.close()
        return

    async def _ensure_index(self, vector_dims: int | None = None) -> None:
        if await self._index_exists():
            has_metadata = await self._async_load_persisted_view_if_present()
            self._validate_vector_dims(vector_dims)
            if not has_metadata:
                if self.config.retrieval_mode == "dense":
                    self.vector_dims = vector_dims
                await self._async_write_metadata_doc()
            return
        if self.config.retrieval_mode == "dense":
            if vector_dims is None or vector_dims <= 0:
                raise ValueError("Dense ElasticBackend requires positive vector dims.")
            self.vector_dims = vector_dims
        await self.client.indices.create(
            index=self.config.index_name,
            body={
                "settings": {
                    "number_of_shards": self.config.number_of_shards,
                    "number_of_replicas": self.config.number_of_replicas,
                },
                "mappings": {
                    "properties": self._mapping_properties(vector_dims=vector_dims)
                },
            },
        )
        await self._async_write_metadata_doc()
        return

    async def _async_load_persisted_view_if_present(self) -> bool:
        if not await self._index_exists():
            return False
        response = await self.client.search(
            index=self.config.index_name,
            body={
                "query": {"term": {INTERNAL_CONTEXT_ID: INTERNAL_META_CONTEXT_ID}},
                "size": 1,
            },
        )
        hits = self._body(response).get("hits", {}).get("hits", [])
        if not hits:
            return False
        source = hits[0].get("_source", {})
        self._load_persisted_view(source.get(INTERNAL_VIEW_DEFINITION))
        self._load_persisted_schema(source.get(INTERNAL_BACKEND_SCHEMA))
        return INTERNAL_VIEW_DEFINITION in source

    async def _async_write_metadata_doc(self) -> None:
        view = self._require_view()
        await self._bulk(
            [
                {
                    "index": {
                        "_index": self.config.index_name,
                        "_id": INTERNAL_META_DOC_ID,
                    }
                },
                {
                    INTERNAL_CONTEXT_ID: INTERNAL_META_CONTEXT_ID,
                    INTERNAL_VIEW_DEFINITION: view.to_dict(),
                    INTERNAL_BACKEND_SCHEMA: self._schema_dict(),
                },
            ]
        )
        return

    async def _index_exists(self) -> bool:
        response = await self.client.indices.exists(index=self.config.index_name)
        body = self._body(response)
        if isinstance(body, bool):
            return body
        return bool(response)

    async def _bulk(self, operations: list[dict[str, Any]]) -> None:
        response = await self.client.bulk(
            operations=operations,
            index=self.config.index_name,
        )
        body = self._body(response)
        if body.get("errors"):
            failed = [
                item.get("index", {}).get("_id")
                for item in body.get("items", [])
                if item.get("index", {}).get("status", 500) >= 300
            ]
            raise RuntimeError(f"Failed to index Elastic rows: {failed}")
        return

    def _merge_response_hits(self, response: dict[str, Any], top_k: int) -> list[Hit]:
        if response.get("status", 200) >= 300:
            return []
        view = self._require_view()
        by_context: dict[str, list[float]] = defaultdict(list)
        contexts: dict[str, Context] = {}
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            context_id = source.get(INTERNAL_CONTEXT_ID)
            if context_id is None or context_id == INTERNAL_META_CONTEXT_ID:
                continue
            by_context[str(context_id)].append(float(hit.get("_score") or 0.0))
            payload = source.get(INTERNAL_PAYLOAD)
            if payload is not None:
                contexts[str(context_id)] = payload_to_context(payload)
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

    def _mapping_properties(
        self,
        *,
        vector_dims: int | None = None,
    ) -> dict[str, Any]:
        properties: dict[str, Any] = {
            INTERNAL_CONTEXT_ID: {"type": "keyword"},
            INTERNAL_VIEW: {"type": "keyword"},
            INTERNAL_FIELD: {"type": "keyword"},
            INTERNAL_PAYLOAD: {"enabled": False},
        }
        if self.config.retrieval_mode == "sparse":
            properties[INTERNAL_TEXT] = {"type": "text"}
            return properties
        if vector_dims is None:
            raise ValueError("Dense ElasticBackend mapping requires vector dims.")
        vector_mapping: dict[str, Any] = {
            "type": "dense_vector",
            "dims": vector_dims,
            "index": True,
            "similarity": self.config.vector_similarity,
        }
        if self.config.index_options is not None:
            vector_mapping["index_options"] = dict(self.config.index_options)
        properties[INTERNAL_VECTOR] = vector_mapping
        return properties

    def _build_sparse_search_body(
        self,
        query: str,
        row_k: int,
        search_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options = self._merged_search_options(search_options)
        forbidden = {"query", "size"}
        conflicts = forbidden.intersection(options)
        if conflicts:
            raise ValueError(
                "ElasticBackend sparse search_options cannot override "
                f"{sorted(conflicts)}."
            )
        body = {
            "query": {
                "bool": {
                    "must": [{"match": {INTERNAL_TEXT: query}}],
                    "filter": [{"term": {INTERNAL_VIEW: self._require_view().name}}],
                }
            },
            "size": row_k,
        }
        body.update(options)
        return body

    def _build_dense_search_body(
        self,
        vector: list[float],
        row_k: int,
        *,
        search_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options = self._merged_search_options(search_options)
        forbidden = {"field", "query_vector", "k"}
        conflicts = forbidden.intersection(options)
        if conflicts:
            raise ValueError(
                "ElasticBackend dense search_options cannot override "
                f"{sorted(conflicts)}."
            )
        user_filter = options.pop("filter", None)
        filters = [{"term": {INTERNAL_VIEW: self._require_view().name}}]
        if isinstance(user_filter, list):
            filters.extend(user_filter)
        elif user_filter is not None:
            filters.append(user_filter)
        knn = {
            **options,
            "field": INTERNAL_VECTOR,
            "query_vector": vector,
            "k": row_k,
            "filter": filters,
        }
        return {
            "knn": knn,
            "size": row_k,
        }

    def _merged_search_options(
        self,
        search_options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        options = dict(self.config.search_options or {})
        options.update(search_options or {})
        return options

    def _schema_dict(self) -> dict[str, Any]:
        if self.config.retrieval_mode == "sparse":
            return {
                "retrieval_mode": "sparse",
                "vector_dims": None,
                "vector_similarity": None,
                "index_options": None,
            }
        return {
            "retrieval_mode": "dense",
            "vector_dims": self.vector_dims,
            "vector_similarity": self.config.vector_similarity,
            "index_options": self.config.index_options,
        }

    def _load_persisted_schema(self, payload: dict[str, Any] | None) -> None:
        if payload is None:
            return
        mode = payload.get("retrieval_mode")
        if mode != self.config.retrieval_mode:
            raise ValueError(
                "ElasticBackend retrieval_mode does not match persisted schema."
            )
        if mode == "sparse":
            return
        if payload.get("vector_similarity") != self.config.vector_similarity:
            raise ValueError(
                "ElasticBackend vector_similarity does not match persisted schema."
            )
        persisted_options = payload.get("index_options")
        if persisted_options != self.config.index_options:
            raise ValueError(
                "ElasticBackend index_options do not match persisted schema."
            )
        vector_dims = payload.get("vector_dims")
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
                "Encoded vector size does not match the persisted Elastic schema."
            )
        return

    def _row_id(self, context_id: str, field: str) -> str:
        return f"{self._require_view().name}:{context_id}:{field}"

    @staticmethod
    def _body(response: Any) -> Any:
        return response.body if hasattr(response, "body") else response
