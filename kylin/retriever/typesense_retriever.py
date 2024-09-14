import logging
from dataclasses import dataclass
from typing import Generator, Iterable

from omegaconf import MISSING

from kylin.utils import Choices, SimpleProgressLogger

from .fingerprint import Fingerprint
from .keyword import Keywords
from .retriever_base import LocalRetriever, LocalRetrieverConfig, RetrievedContext

logger = logging.getLogger("TypesenseRetriever")


@dataclass
class TypesenseRetrieverConfig(LocalRetrieverConfig):
    host: str = MISSING
    port: int = 8108
    protocol: Choices(["https", "http"]) = "http"  # type: ignore
    api_key: str = MISSING
    source: str = MISSING


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
                "connection_timeout_seconds": 2,
            }
        )
        self.source = cfg.source
        if cfg.source not in self._sources:
            schema = self._prepare_schema()
            self.client.collections.create(schema)

        # prepare fingerprint
        self._fingerprint = Fingerprint(
            features={
                "host": cfg.host,
                "port": cfg.port,
                "protocol": cfg.protocol,
                "api_key": cfg.api_key,
                "source": cfg.source,
            }
        )
        return

    def _prepare_schema(self) -> None:
        return {
            "name": self.source,
            "fields": [
                {"name": "title", "type": "string"},
                {"name": "section", "type": "string"},
                {"name": "text", "type": "string"},
            ],
        }

    def add_passages(self, passages: Iterable[dict[str, str]]) -> None:

        def get_batch() -> Generator[list[dict[str, str]]]:
            batch = []
            for passage in passages:
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                batch.append(passage)
            if batch:
                yield batch
            return

        p_logger = SimpleProgressLogger(
            logger=logger,
            total=len(passages) if isinstance(passages, list) else None,
            interval=self.log_interval,
        )
        for batch in get_batch():
            texts = [i["text"] for i in batch]
            r = self.client.collections[self.source].documents.import_(batch)
            assert all([i["success"] for i in r])
            self._fingerprint.update(texts)
            p_logger.update(self.batch_size, desc="Adding passages")
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        search_method = search_kwargs.get("search_method", "full_text")
        match search_method:
            case "full_text":
                results = self._full_text_search(
                    query,
                    top_k,
                    **search_kwargs,
                )
            case "keyword":
                results = self._keywords_search(
                    query,
                    top_k,
                    **search_kwargs,
                )
            case _:
                raise ValueError(f"Unknown search method: {search_method}")
        return results

    def _full_text_search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        search_params = [
            {
                "collection": self.source,
                "q": q,
                "query_by": "title,section,text",
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
                    text=i["document"]["text"],
                    full_text=i["document"]["text"],
                    title=i["document"]["title"],
                    section=i["document"]["section"],
                    source=self.source,
                )
                for i in response["hits"]
            ]
            for q, response in zip(query, responses["results"])
        ]
        return retrieved

    def _keywords_search(
        self,
        query: list[Keywords],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        search_params = []
        for kws in query:
            # convert keywords to typesense query
            filter_by = []
            q = []
            for keyword in kws:
                if keyword.must or (keyword.weight >= 2):
                    filter_by.append(f"text: {keyword.keyword}")
                elif keyword.must_not:
                    filter_by.append(f"text:! {keyword.keyword}")
                else:
                    q.append(keyword.keyword)
            if len(q) == 0:
                q = ["*"]
            search_params.append(
                {
                    "collection": self.source,
                    "q": " ".join(q),
                    "query_by": "title,section,text",
                    "filter_by": " && ".join(filter_by),
                    "per_page": top_k,
                    **search_kwargs,
                }
            )
        responses = self.client.multi_search.perform(
            search_queries={"searches": search_params},
            common_params={},
        )
        retrieved = [
            [
                RetrievedContext(
                    retriever="Typesense",
                    query=str(q),
                    text=i["document"]["text"],
                    full_text=i["document"]["text"],
                    title=i["document"]["title"],
                    section=i["document"]["section"],
                    source=self.source,
                )
                for i in response["hits"]
            ]
            for q, response in zip(query, responses["results"])
        ]
        return retrieved

    def clean(self) -> None:
        if self.source in self._sources:
            self.client.collections[self.source].delete()
        schema = self._prepare_schema()
        self.client.collections.create(schema)
        return

    def close(self) -> None:
        return

    def __len__(self) -> int:
        info = self.client.collections.retrieve()
        info = [i for i in info if i["name"] == self.source][0]
        return info["num_documents"]

    @property
    def fingerprint(self) -> str:
        return self._fingerprint.hexdigest()

    @property
    def _sources(self) -> list[str]:
        return [i["name"] for i in self.client.collections.retrieve()]
