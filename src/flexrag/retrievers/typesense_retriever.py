import asyncio
from typing import Annotated, Iterable, Optional

from flexrag.common import (
    LOGGER_MANAGER,
    Choices,
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

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.typesense")


@configure
class TypesenseRetrieverConfig(RetrieverBaseConfig):
    """Configuration class for TypesenseRetriever.

    :param host: Host of the Typesense server. Default: "localhost".
    :type host: str
    :param port: Port of the Typesense server. Default: 8108.
    :type port: int
    :param protocol: Protocol of the Typesense server. Default: "http".
        Available options: "https", "http".
    :type protocol: str
    :param api_key: API key for the Typesense server. Required.
    :type api_key: str
    :param index_name: Name of the Typesense collection. Required.
    :type index_name: str
    :param timeout: Timeout for the connection. Default: 200.0.
    :type timeout: float
    """

    host: str = "localhost"
    port: int = 8108
    protocol: Annotated[str, Choices("https", "http")] = "http"
    api_key: Optional[str] = None
    index_name: Optional[str] = None
    timeout: float = 200.0


@RETRIEVERS("typesense", config_class=TypesenseRetrieverConfig)
class TypesenseRetriever(RemoteRetrieverBase):
    def __init__(self, cfg: TypesenseRetrieverConfig) -> None:
        super().__init__(cfg)
        self.cfg = extract_config(cfg, TypesenseRetrieverConfig)
        assert self.cfg.api_key is not None, "`api_key` must be provided"
        assert self.cfg.index_name is not None, "`index_name` must be provided"

        # load database
        try:
            import typesense
        except ImportError as exc:
            raise ImportError(
                "Please install Typesense support by running "
                "`pip install 'flexrag[typesense]'`."
            ) from exc

        self.typesense = typesense
        self.client = typesense.AsyncClient(
            {
                "nodes": [
                    {
                        "host": cfg.host,
                        "port": cfg.port,
                        "protocol": cfg.protocol,
                    }
                ],
                "api_key": cfg.api_key,
                "connection_timeout_seconds": cfg.timeout,
            }
        )
        self.index_name = cfg.index_name
        return

    @trace("retriever.typesense.add_passages")
    async def async_add_passages(
        self,
        passages: Iterable[Context],
        log_interval: int = 10000,
        display: ProgressDisplay = "auto",
    ) -> None:
        # create collection if not exists
        if self.index_name not in await self.async_indices():
            schema = {
                "name": self.index_name,
                "fields": [
                    {"name": ".*", "type": "auto", "index": True, "infix": True}
                ],
            }
            await self.client.collections.create(schema)

        # import documents
        batch = []
        with SimpleProgressLogger(
            logger, interval=log_interval, display=display
        ) as p_logger:
            for passage in passages:
                if len(batch) == self.cfg.batch_size:
                    r = await self.client.collections[
                        self.index_name
                    ].documents.import_(batch)
                    assert all([i["success"] for i in r])
                    p_logger.update(len(batch), desc="Adding passages")
                    batch = []
                data = passage.data.copy()
                data[self.id_field_name] = passage.context_id
                batch.append(data)
            if batch:
                r = await self.client.collections[
                    self.index_name
                ].documents.import_(batch)
                assert all([i["success"] for i in r])
                p_logger.update(len(batch), desc="Adding passages")
        logger.info("Finished adding passages.")
        return

    @trace("retriever.typesense.search")
    async def _async_search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # prepare search parameters
        top_k = search_kwargs.pop("top_k", DEFAULT_TOP_K)
        fields = await self.async_fields()
        search_params = [
            {
                "collection": self.index_name,
                "q": q,
                "query_by": ",".join(fields),
                "per_page": top_k,
                **search_kwargs,
            }
            for q in query
        ]

        # search
        try:
            responses = await self.client.multi_search.perform(
                search_queries={"searches": search_params},
                common_params={},
            )
        except self.typesense.exceptions.TypesenseClientError as e:
            logger.error(f"Typesense error: {e}")
            logger.error(f"Current query: {query}")
            return [[] for _ in query]

        # form final results
        retrieved = []
        for q, response in zip(query, responses["results"]):
            retrieved.append([])
            for i in response["hits"]:
                data = i["document"].copy()
                context_id = data.pop(self.id_field_name)
                retrieved[-1].append(
                    RetrievedContext(
                        context_id=context_id,
                        retriever="Typesense",
                        query=q,
                        data=data,
                        source=self.index_name,
                        score=i["text_match"],
                    )
                )
        return retrieved

    async def async_get(self, context_id: str) -> Context:
        try:
            res = await (
                self.client.collections[self.index_name]
                .documents[context_id]
                .retrieve()
            )
            data = res.copy()
            c_id = data.pop(self.id_field_name)
            return Context(
                context_id=c_id,
                data=data,
                source=self.index_name,
            )
        except self.typesense.exceptions.ObjectNotFound:
            raise KeyError(context_id)

    async def async_clear(self) -> None:
        if self.index_name in await self.async_indices():
            await self.client.collections[self.index_name].delete()
        return

    async def async_count(self) -> int:
        info = await self.client.collections.retrieve()
        info = [i for i in info if i["name"] == self.index_name]
        if len(info) > 0:
            return info[0]["num_documents"]
        return 0

    async def async_indices(self) -> list[str]:
        return [i["name"] for i in await self.client.collections.retrieve()]

    @property
    def indices(self) -> list[str]:
        self._ensure_sync_bridge_allowed("indices")
        return asyncio.run(self.async_indices())

    async def async_fields(self) -> list[str]:
        return [
            i["name"]
            for i in (await self.client.collections[self.index_name].retrieve())[
                "fields"
            ]
            if i["name"] != ".*"
        ]

    @property
    def id_field_name(self) -> str:
        return "id"  # `id` is the reserved field name in Typesense
