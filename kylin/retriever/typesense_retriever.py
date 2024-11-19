import logging
from dataclasses import dataclass
from typing import Generator, Iterable

from omegaconf import MISSING

from kylin.utils import Choices, SimpleProgressLogger

from .keyword import Keywords
from .retriever_base import (
    SPARSE_RETRIEVERS,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrievedContext,
)

logger = logging.getLogger("TypesenseRetriever")


@dataclass
class TypesenseRetrieverConfig(LocalRetrieverConfig):
    host: str = MISSING
    port: int = 8108
    protocol: Choices(["https", "http"]) = "http"  # type: ignore
    api_key: str = MISSING
    source: str = MISSING
    timeout: float = 200.0


@SPARSE_RETRIEVERS("typesense", config_class=TypesenseRetrieverConfig)
class TypesenseRetriever(LocalRetriever):
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
        self.source = cfg.source
        return

    def add_passages(self, passages: Iterable[dict[str, str]]) -> None:
        def get_batch() -> Generator[list[dict[str, str]], None, None]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                batch.append(passage)
            if batch:
                yield batch
            return

        # create collection if not exists
        if self.source not in self._sources:
            schema = {
                "name": self.source,
                "fields": [{"name": ".*", "type": "auto"}],
            }
            self.client.collections.create(schema)

        # import documents
        p_logger = SimpleProgressLogger(logger=logger, interval=self.log_interval)
        for batch in get_batch():
            r = self.client.collections[self.source].documents.import_(batch)
            assert all([i["success"] for i in r])
            p_logger.update(len(batch), desc="Adding passages")
        return

    def search_batch(
        self,
        query: list[str | Keywords],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        search_params = [
            {
                "collection": self.source,
                "q": q,
                "query_by": ",".join(self.fields),
                "per_page": top_k,
                **search_kwargs,
            }
            for q in query
        ]
        responses = self.client.multi_search.perform(
            search_queries={"searches": search_params},
            common_params={},
        )
        retrieved = [
            [
                RetrievedContext(
                    retriever="Typesense",
                    query=q,
                    data=i["document"],
                    source=self.source,
                    score=i["text_match"],
                )
                for i in response["hits"]
            ]
            for q, response in zip(query, responses["results"])
        ]
        return retrieved

    def clean(self) -> None:
        if self.source in self._sources:
            self.client.collections[self.source].delete()
        return

    def close(self) -> None:
        return

    def __len__(self) -> int:
        info = self.client.collections.retrieve()
        info = [i for i in info if i["name"] == self.source][0]
        return info["num_documents"]

    @property
    def _sources(self) -> list[str]:
        return [i["name"] for i in self.client.collections.retrieve()]

    @property
    def fields(self) -> list[str]:
        return [
            i["name"]
            for i in self.client.collections[self.source].retrieve()["fields"]
            if i["name"] != ".*"
        ]
