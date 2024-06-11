import time
import logging
from argparse import ArgumentParser, Namespace
from typing import Iterable

from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk

from .retriever_base import LocalRetriever


class BM25Retriever(LocalRetriever):
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
        logger = logging.getLogger("elastic_transport.transport")
        if self.verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)
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

    def _search_batch(
        self, query: list[str], top_k: int = 10, **search_kwargs
    ) -> list[dict[str, str | list]]:
        results = []
        for q in query:
            res = self.client.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": q,
                            "fields": ["title", "section", "text"],
                        }
                    }
                },
                size=top_k,
            )["hits"]["hits"]
            res = {
                "query": q,
                "indices": [i["_id"] for i in res],
                "scores": [i["_score"] for i in res],
                "titles": [i["_source"]["title"] for i in res],
                "sections": [i["_source"]["section"] for i in res],
                "texts": [i["_source"]["text"] for i in res],
            }
            results.append(res)
        return results

    def close(self) -> None:
        self.client.close()
        return

    def __len__(self) -> int:
        return self.client.count(index=self.index_name)["count"]
