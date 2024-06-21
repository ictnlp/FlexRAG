import logging
import re
import time
from argparse import ArgumentParser, Namespace
from typing import Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from tenacity import retry, stop_after_attempt

from .retriever_base import LocalRetriever


logger = logging.getLogger("BM25Retriever")


class BM25Retriever(LocalRetriever):
    name = "BM25"

    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--host",
            type=str,
            default="http://localhost:9200/",
            help="The host of the Elasticsearch server",
        )
        parser.add_argument(
            "--api_key",
            type=str,
            default=None,
            help="The api key for the Elasticsearch server",
        )
        parser.add_argument(
            "--index_name",
            type=str,
            default="documents",
            help="The name of the index to use for the BM25 retriever",
        )
        parser.add_argument(
            "--elastic_verbose",
            default=False,
            action="store_true",
            help="Whether to show verbose log output from elasticsearch",
        )
        return parser

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.host = args.host
        self.api_key = args.api_key
        self.index_name = args.index_name
        self.verbose = args.elastic_verbose
        self._prep_client()
        return

    def _prep_client(self):
        mapping = {
            "properties": {
                "title": {"type": "text", "analyzer": "english"},
                "section": {"type": "text", "analyzer": "english"},
                "text": {"type": "text", "analyzer": "english"},
            }
        }
        # set client
        self.client = Elasticsearch(
            self.host,
            api_key=self.api_key,
        )
        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(
                index=self.index_name,
                mappings=mapping,
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
        source: str = None,
        reinit: bool = False,
    ):
        if reinit:
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
    ) -> list[dict[str, str | list]]:
        # prepare retry
        if retry_times > 1:
            search_method = retry(stop=stop_after_attempt(retry_times))(
                self.client.msearch
            )
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
        res = search_method(body=body)["responses"]
        results = []
        for r, q in zip(res, query):
            r = r["hits"]["hits"]
            results.append(
                {
                    "query": q,
                    "indices": [i["_id"] for i in r],
                    "scores": [i["_score"] for i in r],
                    "titles": [i["_source"]["title"] for i in r],
                    "sections": [i["_source"]["section"] for i in r],
                    "texts": [i["_source"]["text"] for i in r],
                }
            )
        return results

    def _string_search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        # prepare retry
        if retry_times > 1:
            search_method = retry(stop=stop_after_attempt(retry_times))(
                self.client.msearch
            )
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
        res = search_method(body=body)["responses"]
        results = []
        for r, q in zip(res, query):
            r = r["hits"]["hits"]
            results.append(
                {
                    "query": q,
                    "indices": [i["_id"] for i in r],
                    "scores": [i["_score"] for i in r],
                    "titles": [i["_source"]["title"] for i in r],
                    "sections": [i["_source"]["section"] for i in r],
                    "texts": [i["_source"]["text"] for i in r],
                }
            )
        return results

    def _bool_search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> dict[str, dict | list]:

        # TODO: Finish this method
        def parse_query(query: str) -> tuple[list[str], list[str], list[str]]:
            phrases = re.findall(r'(?:"([^"]+)"|(\S+))', query)
            must = [p[0] for p in phrases if p[0]]
            must_not = [p[1] for p in phrases if p[1] and p[1].startswith("-")]
            should = [p[1] for p in phrases if p[1] and not p[1].startswith("-")]
            raise NotImplementedError

        musts = []
        must_nots = []
        shoulds = []
        for q in query:
            must, must_not, should = parse_query(q)
            musts.append(must)
            must_nots.append(must_not)
            shoulds.append(should)
        responses = self._advanced_search(
            musts=musts,
            must_nots=must_nots,
            shoulds=shoulds,
            top_k=top_k,
            retry_times=retry_times,
            **search_kwargs,
        )
        results = []
        for r, q in zip(responses, query):
            r = r["hits"]["hits"]
            results.append(
                {
                    "query": q,
                    "indices": [i["_id"] for i in r],
                    "scores": [i["_score"] for i in r],
                    "titles": [i["_source"]["title"] for i in r],
                    "sections": [i["_source"]["section"] for i in r],
                    "texts": [i["_source"]["text"] for i in r],
                }
            )
        return results

    def _mutex_search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> dict[str, dict | list]:
        def parse_query(query: str) -> tuple[list[str], str]:
            must_not = re.findall(r'<([^>]+)>', query)
            should = re.sub(r'<[^>]+>', "", query)
            return must_not, should

        must_nots = []
        shoulds = []
        for q in query:
            must_not, should = parse_query(q)
            must_nots.append(must_not)
            shoulds.append([should])
        responses = self._advanced_search(
            musts=[[] for _ in query],
            must_nots=must_nots,
            shoulds=shoulds,
            top_k=top_k,
            retry_times=retry_times,
            **search_kwargs,
        )
        results = []
        for r, q in zip(responses, query):
            r = r["hits"]["hits"]
            results.append(
                {
                    "query": query,
                    "indices": [i["_id"] for i in r],
                    "scores": [i["_score"] for i in r],
                    "titles": [i["_source"]["title"] for i in r],
                    "sections": [i["_source"]["section"] for i in r],
                    "texts": [i["_source"]["text"] for i in r],
                }
            )
        return results

    def _advanced_search(
        self,
        musts: list[list[str]],
        must_nots: list[list[str]],
        shoulds: list[list[str]],
        top_k: int = 10,
        retry_times: int = 3,
        minimum_should_match: int = 0,
        **search_kwargs,
    ) -> dict[str, dict | list]:
        # prepare retry
        if retry_times > 1:
            search_method = retry(stop=stop_after_attempt(retry_times))(
                self.client.msearch
            )
        else:
            search_method = self.client.msearch

        # parse search clause
        body = []
        for must, must_not, should in zip(musts, must_nots, shoulds):
            must_clause = [{"match": {"text": m}} for m in must]
            should_clause = [{"match": {"text": s}} for s in should]
            must_not_clause = [{"match_phrase": {"text": mn}} for mn in must_not]
            body.append({"index": self.index_name})
            body.append(
                {
                    "query": {
                        "bool": {
                            "must": must_clause,
                            "should": should_clause,
                            "must_not": must_not_clause,
                            "minimum_should_match": minimum_should_match,
                        },
                    },
                    "size": top_k,
                }
            )

        # search
        return search_method(body=body)["responses"]

    def _search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        retry_times: int = 3,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        search_method = search_kwargs.get("search_method", "string")
        match search_method:
            case "full_text":
                results = self._full_text_search(
                    query=query,
                    top_k=top_k,
                    retry_times=retry_times,
                    **search_kwargs,
                )
            case "bool":
                results = self._bool_search(
                    query=query,
                    top_k=top_k,
                    retry_times=retry_times,
                    **search_kwargs,
                )
            case "mutex":
                results = self._mutex_search(
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
