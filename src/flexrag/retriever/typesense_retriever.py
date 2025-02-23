from dataclasses import dataclass
from typing import Generator, Iterable

from omegaconf import MISSING

from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.utils import LOGGER_MANAGER, TIME_METER, Choices, SimpleProgressLogger

from .retriever_base import RETRIEVERS, EditableRetriever, EditableRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.typesense")


@dataclass
class TypesenseRetrieverConfig(EditableRetrieverConfig):
    """Configuration class for TypesenseRetriever.

    :param host: Host of the Typesense server. Required.
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

    host: str = MISSING
    port: int = 8108
    protocol: Choices(["https", "http"]) = "http"  # type: ignore
    api_key: str = MISSING
    index_name: str = MISSING
    timeout: float = 200.0


@RETRIEVERS("typesense", config_class=TypesenseRetrieverConfig)
class TypesenseRetriever(EditableRetriever):
    name = "Typesense"

    def __init__(self, cfg: TypesenseRetrieverConfig) -> None:
        super().__init__(cfg)
        import typesense

        # load database
        self.typesense = typesense
        self.client = typesense.Client(
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

    @TIME_METER("typesense", "add_passages")
    def add_passages(self, passages: Iterable[Context]) -> None:
        def get_batch() -> Generator[list[dict[str, str]], None, None]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                data = passage.data.copy()
                data[self.id_field_name] = passage.context_id
                batch.append(data)
            if batch:
                yield batch
            return

        # create collection if not exists
        if self.index_name not in self.indices:
            schema = {
                "name": self.index_name,
                "fields": [
                    {"name": ".*", "type": "auto", "index": True, "infix": True}
                ],
            }
            self.client.collections.create(schema)

        # import documents
        p_logger = SimpleProgressLogger(logger=logger, interval=self.log_interval)
        for batch in get_batch():
            r = self.client.collections[self.index_name].documents.import_(batch)
            assert all([i["success"] for i in r])
            p_logger.update(len(batch), desc="Adding passages")
        logger.info("Finished adding passages.")
        return

    @TIME_METER("typesense", "search")
    def search_batch(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        # prepare search parameters
        search_params = [
            {
                "collection": self.index_name,
                "q": q,
                "query_by": ",".join(self.fields),
                "per_page": search_kwargs.pop("top_k", self.top_k),
                **search_kwargs,
            }
            for q in query
        ]

        # search
        try:
            responses = self.client.multi_search.perform(
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
                data = i["document"]
                context_id = data.pop(self.id_field_name)
                retrieved[-1].append(
                    RetrievedContext(
                        context_id=context_id,
                        retriever="Typesense",
                        query=q,
                        data=i["document"],
                        source=self.index_name,
                        score=i["text_match"],
                    )
                )
        return retrieved

    def clean(self) -> None:
        if self.index_name in self.indices:
            self.client.collections[self.index_name].delete()
        return

    def __len__(self) -> int:
        info = self.client.collections.retrieve()
        info = [i for i in info if i["name"] == self.index_name]
        if len(info) > 0:
            return info[0]["num_documents"]
        return 0

    @property
    def indices(self) -> list[str]:
        return [i["name"] for i in self.client.collections.retrieve()]

    @property
    def fields(self) -> list[str]:
        return [
            i["name"]
            for i in self.client.collections[self.index_name].retrieve()["fields"]
            if i["name"] != ".*"
        ]

    @property
    def id_field_name(self) -> str:
        return "id"  # `id` is the reserved field name in Typesense
