import asyncio
import logging
from typing import Any, Iterable, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError

from flexrag.common import (
    LOGGER_MANAGER,
    ProgressDisplay,
    SimpleProgressLogger,
    configure,
    trace,
)
from flexrag.common.configure import extract_config
from flexrag.common.dataclasses import Context, RetrievedContext

from .retriever_base import (
    DEFAULT_TOP_K,
    RETRIEVERS,
    RemoteRetrieverBase,
    RetrieverBaseConfig,
)

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.elastic")


@configure
class ElasticRetrieverConfig(RetrieverBaseConfig):
    """Configuration class for ElasticRetriever.

    :param host: Host of the ElasticSearch server. Default: "http://localhost:9200".
    :type host: str
    :param api_key: API key for the ElasticSearch server. Default: None.
    :type api_key: Optional[str]
    :param index_name: Name of the index. Required.
    :type index_name: str
    :param custom_properties: Custom properties for building the index. Default: None.
    :type custom_properties: Optional[dict]
    :param verbose: Enable verbose logging mode. Default: False.
    :type verbose: bool
    :param retry_times: Number of retry times. Default: 3.
    :type retry_times: int
    :param retry_delay: Delay time for retry. Default: 0.5.
    :type retry_delay: float
    """

    host: str = "http://localhost:9200"
    api_key: Optional[str] = None
    index_name: Optional[str] = None
    custom_properties: Optional[dict] = None
    verbose: bool = False
    retry_times: int = 3
    retry_delay: float = 0.5


@RETRIEVERS("elastic", config_class=ElasticRetrieverConfig)
class ElasticRetriever(RemoteRetrieverBase):
    name = "ElasticSearch"

    def __init__(self, cfg: ElasticRetrieverConfig) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, ElasticRetrieverConfig)
        # set basic args
        self.host = cfg.host
        self.api_key = cfg.api_key
        assert cfg.index_name is not None, "`index_name` must be provided"
        self.index_name = cfg.index_name
        self.verbose = cfg.verbose
        self.retry_times = cfg.retry_times
        self.retry_delay = cfg.retry_delay
        self.custom_properties = cfg.custom_properties

        # prepare client
        self.client = AsyncElasticsearch(
            self.host,
            api_key=self.api_key,
            max_retries=cfg.retry_times,
            retry_on_timeout=True,
        )

        # set logger
        transport_logger = logging.getLogger("elastic_transport.transport")
        es_logger = logging.getLogger("elasticsearch")
        if self.verbose:
            transport_logger.setLevel(logging.INFO)
            es_logger.setLevel(logging.INFO)
        else:
            transport_logger.setLevel(logging.WARNING)
            es_logger.setLevel(logging.WARNING)
        return

    @staticmethod
    def _response_body(response: Any) -> Any:
        return response.body if hasattr(response, "body") else response

    @classmethod
    def _response_bool(cls, response: Any) -> bool:
        body = cls._response_body(response)
        if isinstance(body, bool):
            return body
        return bool(response)

    @trace("retriever.elastic_search.add_passages")
    async def async_add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        index_exists = self._response_bool(
            await self.client.indices.exists(index=self.index_name)
        )
        actions = []

        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for passage in passages:
                if not index_exists:
                    properties = (
                        {
                            key: {"type": "text", "analyzer": "english"}
                            for key in passage.data.keys()
                        }
                        if self.custom_properties is None
                        else self.custom_properties
                    )
                    index_body = {
                        "settings": {"number_of_shards": 1, "number_of_replicas": 1},
                        "mappings": {"properties": properties},
                    }
                    await self.client.indices.create(
                        index=self.index_name,
                        body=index_body,
                    )
                    index_exists = True

                action = {
                    "index": {
                        "_index": self.index_name,
                        "_id": passage.context_id,
                    }
                }
                actions.append(action)
                actions.append(passage.data)
                if len(actions) == self.cfg.batch_size * 2:
                    r = await self.client.bulk(
                        operations=actions,
                        index=self.index_name,
                    )
                    body = self._response_body(r)
                    if body["errors"]:
                        err_passage_ids = [
                            item["index"]["_id"]
                            for item in body["items"]
                            if item["index"]["status"] != 201
                        ]
                        raise RuntimeError(
                            f"Failed to index passages: {err_passage_ids}"
                        )
                    p_logger.update(len(actions) // 2, "Indexing")
                    actions = []

            if actions:
                r = await self.client.bulk(
                    operations=actions,
                    index=self.index_name,
                )
                body = self._response_body(r)
                if body["errors"]:
                    err_passage_ids = [
                        item["index"]["_id"]
                        for item in body["items"]
                        if item["index"]["status"] != 201
                    ]
                    raise RuntimeError(f"Failed to index passages: {err_passage_ids}")
                p_logger.update(len(actions) // 2, "Indexing")
        logger.info(f"Finished adding passages.")
        return

    @trace("retriever.elastic_search.search")
    async def _async_search(
        self,
        query: list[str],
        search_method: str = "full_text",
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # check search method
        match search_method:
            case "full_text":
                query_type = "multi_match"
            case "lucene":
                query_type = "query_string"
            case _:
                raise ValueError(f"Invalid search method: {search_method}")

        # prepare search body
        body = []
        top_k = search_kwargs.pop("top_k", DEFAULT_TOP_K)
        fields = await self.async_fields()
        for q in query:
            body.append({"index": self.index_name})
            body.append(
                {
                    "query": {
                        query_type: {
                            "query": q,
                            "fields": fields,
                        },
                    },
                    "size": top_k,
                }
            )

        # search and post-process
        response = await self.client.msearch(body=body, **search_kwargs)
        responses = self._response_body(response)["responses"]
        return self._form_results(query, responses)

    async def async_clear(self) -> None:
        if self.index_name in await self.async_indices():
            await self.client.indices.delete(index=self.index_name)
        return

    async def async_count(self) -> int:
        if self.index_name in await self.async_indices():
            response = await self.client.count(index=self.index_name)
            return self._response_body(response)["count"]
        return 0

    async def async_indices(self) -> list[str]:
        response = await self.client.cat.indices(format="json")
        return [i["index"] for i in self._response_body(response)]

    @property
    def indices(self) -> list[str]:
        self._ensure_sync_bridge_allowed("indices")
        return asyncio.run(self.async_indices())

    def _form_results(
        self, query: list[str], responses: list[dict] | None
    ) -> list[list[RetrievedContext]]:
        results = []
        if responses is None:
            responses = [{"status": 500}] * len(query)
        for r, q in zip(responses, query):
            if r["status"] != 200:
                results.append(
                    [
                        RetrievedContext(
                            retriever=self.name,
                            query=q,
                            data={},
                            source=self.index_name,
                            score=0.0,
                        )
                    ]
                )
                continue
            r = r["hits"]["hits"]
            results.append(
                [
                    RetrievedContext(
                        context_id=i["_id"],
                        retriever=self.name,
                        query=q,
                        data=i["_source"],
                        source=i["_index"],
                        score=i["_score"],
                    )
                    for i in r
                ]
            )
        return results

    async def async_fields(self) -> list[str]:
        if self.index_name in await self.async_indices():
            response = await self.client.indices.get_mapping(index=self.index_name)
            mapping = self._response_body(response)
            return list(mapping[self.index_name]["mappings"]["properties"].keys())
        return []

    async def async_get(self, context_id: str) -> Context:
        try:
            response = await self.client.get(index=self.index_name, id=context_id)
            res = self._response_body(response)
            return Context(
                context_id=res["_id"],
                data=res["_source"],
                source=res["_index"],
            )
        except NotFoundError:
            raise KeyError(context_id)

    async def aclose(self) -> None:
        await self._maybe_await(self.client.close())
        return
