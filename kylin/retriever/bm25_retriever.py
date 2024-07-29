import logging
import re
import time
import json
from dataclasses import dataclass
from typing import Iterable, Optional
from uuid import uuid5, NAMESPACE_OID

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tenacity import retry, stop_after_attempt, RetryCallState

from kylin.utils import Choices

from .retriever_base import LocalRetriever, LocalRetrieverConfig, RetrievedContext


logger = logging.getLogger("BM25Retriever")


def _save_error_state(retry_state: RetryCallState):
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("retry_error_state.json", "w") as f:
        json.dump(args, f)
    return


@dataclass
class BM25RetrieverConfig(LocalRetrieverConfig):
    host: str = "http://localhost:9200"
    api_key: Optional[str] = None
    index_name: str = "documents"
    verbose: bool = False
    search_method: Choices(["full_text", "string"]) = "string"  # type: ignore


class BM25Retriever(LocalRetriever):
    name = "BM25"

    def __init__(self, cfg: BM25RetrieverConfig) -> None:
        super().__init__(cfg)
        self.host = cfg.host
        self.api_key = cfg.api_key
        self.index_name = cfg.index_name
        self.verbose = cfg.verbose
        self.search_method = cfg.search_method
        self._prep_client()
        return

    def _prep_client(self):
        # set client
        self.client = Elasticsearch(
            self.host,
            api_key=self.api_key,
        )
        if not self.client.indices.exists(index=self.index_name):
            index_body = {
                "settings": {"number_of_shards": 1, "number_of_replicas": 1},
                "mappings": {
                    "properties": {
                        "title": {"type": "text", "analyzer": "english"},
                        "section": {"type": "text", "analyzer": "english"},
                        "text": {"type": "text", "analyzer": "english"},
                    }
                },
            }
            self.client.indices.create(
                index=self.index_name,
                body=index_body,
            )

        # set logging
        transport_logger = logging.getLogger("elastic_transport.transport")
        es_logger = logging.getLogger("elasticsearch")
        if self.verbose:
            transport_logger.setLevel(logging.INFO)
            es_logger.setLevel(logging.INFO)
        else:
            transport_logger.setLevel(logging.WARNING)
            es_logger.setLevel(logging.WARNING)
        return

    def add_passages(
        self,
        passages: Iterable[dict[str, str]] | list[str],
        reinit: bool = False,
    ):
        if reinit:
            logger.warning("Reinitializing the index")
            self.client.indices.delete(index=self.index_name)
            time.sleep(5)
            self._prep_client()

        def generate_actions():
            for passage in passages:
                p = passage if isinstance(passage, dict) else {"text": passage}
                es_doc = {
                    "_op_type": "index",
                    "refresh": "wait_for",
                    "title": p.get("title", ""),
                    "section": p.get("section", ""),
                    "text": self._prepare_text(p),
                }
                yield es_doc

        for n, (ok, result) in enumerate(
            streaming_bulk(
                client=self.client,
                actions=generate_actions(),
                index=self.index_name,
                chunk_size=self.batch_size,
            )
        ):
            if not ok:
                raise RuntimeError(f"Failed to index passage {n}: {result}")
            if (n / self.batch_size) % self.log_interval == 0:
                print(f"Indexed {n} passages")
        return

    def _full_text_search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        # prepare retry
        if retry_times > 1:
            search_method = retry(
                stop=stop_after_attempt(retry_times),
                retry_error_callback=_save_error_state,
            )(self.client.msearch)
        else:
            search_method = self.client.msearch

        # prepare search body
        body = []
        for q in query:
            body.append({"index": self.index_name})
            body.append(
                {
                    "query": {
                        "multi_match": {
                            "query": q,
                            "fields": ["title", "section", "text"],
                        },
                    },
                    "size": top_k,
                }
            )

        # search and post-process
        responses = search_method(body=body, **search_kwargs)["responses"]
        return self._form_results(query, responses)

    def _string_search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        # prepare retry
        if retry_times > 1:
            search_method = retry(
                stop=stop_after_attempt(retry_times),
                retry_error_callback=_save_error_state,
            )(self.client.msearch)
        else:
            search_method = self.client.msearch

        # prepare search body
        body = []
        for q in query:
            body.append({"index": self.index_name})
            body.append(
                {
                    "query": {
                        "query_string": {
                            "query": q,
                            "fields": ["title", "section", "text"],
                        },
                    },
                    "size": top_k,
                }
            )

        # search and post-process
        responses = search_method(body=body, **search_kwargs)["responses"]
        return self._form_results(query, responses)

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        search_method = search_kwargs.get("search_method", self.search_method)
        match search_method:
            case "full_text":
                results = self._full_text_search(
                    query=query,
                    top_k=top_k,
                    retry_times=retry_times,
                    **search_kwargs,
                )
            case "string":
                results = self._string_search(
                    query=query,
                    top_k=top_k,
                    retry_times=retry_times,
                    **search_kwargs,
                )
            case _:
                raise ValueError(f"Invalid search method: {search_method}")
        return results

    def close(self) -> None:
        self.client.close()
        return

    def __len__(self) -> int:
        return self.client.count(index=self.index_name)["count"]

    @property
    def indices(self) -> list[str]:
        return [i["index"] for i in self.client.cat.indices(format="json")]

    @property
    def fingerprint(self) -> str:
        for index_info in self.client.cat.indices(format="json"):
            if index_info["index"] == self.index_name:
                break
        client_info = self.client.info()
        feature_str = self.host + json.dumps(client_info) + json.dumps(index_info)
        namespace = uuid5(NAMESPACE_OID, self.name)
        return uuid5(namespace, feature_str).hex

    def _form_results(
        self, query: list[str], responses: list
    ) -> list[list[RetrievedContext]]:
        results = []
        for r, q in zip(responses, query):
            if r["status"] != 200:
                results.append(
                    [
                        RetrievedContext(
                            retriever=self.name,
                            query=q,
                            chunk_id="",
                            source=self.index_name,
                            score=0.0,
                            title="",
                            section="",
                            text="",
                            full_text="",
                        )
                    ]
                )
                continue
            r = r["hits"]["hits"]
            results.append(
                [
                    RetrievedContext(
                        retriever=self.name,
                        query=q,
                        chunk_id=i["_id"],
                        source=self.index_name,
                        score=i["_score"],
                        title=i["_source"]["title"],
                        section=i["_source"]["section"],
                        text=i["_source"]["text"],
                        full_text=i["_source"]["text"],
                    )
                    for i in r
                ]
            )
        return results
