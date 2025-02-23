import logging
from dataclasses import dataclass
from typing import Iterable, Optional

from omegaconf import MISSING
from elasticsearch import Elasticsearch

from flexrag.common_dataclass import Context, RetrievedContext
from flexrag.utils import SimpleProgressLogger, LOGGER_MANAGER, TIME_METER

from .retriever_base import RETRIEVERS, EditableRetriever, EditableRetrieverConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.elastic")


@dataclass
class ElasticRetrieverConfig(EditableRetrieverConfig):
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
    index_name: str = MISSING
    custom_properties: Optional[dict] = None
    verbose: bool = False
    retry_times: int = 3
    retry_delay: float = 0.5


@RETRIEVERS("elastic", config_class=ElasticRetrieverConfig)
class ElasticRetriever(EditableRetriever):
    name = "ElasticSearch"

    def __init__(self, cfg: ElasticRetrieverConfig) -> None:
        super().__init__(cfg)
        # set basic args
        self.host = cfg.host
        self.api_key = cfg.api_key
        self.index_name = cfg.index_name
        self.verbose = cfg.verbose
        self.retry_times = cfg.retry_times
        self.retry_delay = cfg.retry_delay
        self.custom_properties = cfg.custom_properties

        # prepare client
        self.client = Elasticsearch(
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

    @TIME_METER("elastic_search", "add_passages")
    def add_passages(self, passages: Iterable[Context]):
        def generate_actions():
            index_exists = self.client.indices.exists(index=self.index_name)
            actions = []
            for n, passage in enumerate(passages):
                # build index if not exists
                if not index_exists:
                    if self.custom_properties is None:
                        properties = {
                            key: {"type": "text", "analyzer": "english"}
                            for key in passage.data.keys()
                        }
                    else:
                        properties = self.custom_properties
                    index_body = {
                        "settings": {"number_of_shards": 1, "number_of_replicas": 1},
                        "mappings": {
                            "properties": properties,
                        },
                    }
                    self.client.indices.create(
                        index=self.index_name,
                        body=index_body,
                    )
                    index_exists = True

                # prepare action
                action = {
                    "index": {
                        "_index": self.index_name,
                        "_id": passage.context_id,
                    }
                }
                actions.append(action)
                actions.append(passage.data)
                if len(actions) == self.batch_size * 2:
                    yield actions
                    actions = []
            if actions:
                yield actions
            return

        p_logger = SimpleProgressLogger(logger, interval=self.log_interval)
        for actions in generate_actions():
            r = self.client.bulk(
                operations=actions,
                index=self.index_name,
            )
            if r.body["errors"]:
                err_passage_ids = [
                    item["index"]["_id"]
                    for item in r.body["items"]
                    if item["index"]["status"] != 201
                ]
                raise RuntimeError(f"Failed to index passages: {err_passage_ids}")
            p_logger.update(len(actions) // 2, "Indexing")
        logger.info(f"Finished adding passages.")
        return

    @TIME_METER("elastic_search", "search")
    def search_batch(
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
        for q in query:
            body.append({"index": self.index_name})
            body.append(
                {
                    "query": {
                        query_type: {
                            "query": q,
                            "fields": self.fields,
                        },
                    },
                    "size": search_kwargs.pop("top_k", self.top_k),
                }
            )

        # search and post-process
        responses = self.client.msearch(body=body, **search_kwargs)["responses"]
        return self._form_results(query, responses)

    def clean(self) -> None:
        if self.index_name in self.indices:
            self.client.indices.delete(index=self.index_name)
        return

    def __len__(self) -> int:
        if self.index_name in self.indices:
            return self.client.count(index=self.index_name)["count"]
        return 0

    @property
    def indices(self) -> list[str]:
        return [i["index"] for i in self.client.cat.indices(format="json")]

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

    @property
    def fields(self) -> list[str]:
        if self.index_name in self.indices:
            mapping = self.client.indices.get_mapping(index=self.index_name)
            return list(mapping[self.index_name]["mappings"]["properties"].keys())
        return []
