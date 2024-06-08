import os
import time
import logging
from abc import abstractmethod
from argparse import ArgumentParser, Namespace

import requests
from tenacity import retry, stop_after_attempt

from .cache import PersistentLRUCache, hashkey
from .retriever_base import Retriever


logger = logging.getLogger(__name__)


class WebRetriever(Retriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--timeout",
            type=float,
            default=1.0,
            help="The timeout for each request",
        )
        parser.add_argument(
            "--cache_size",
            default=0,
            type=int,
            help="The size of the cache",
        )
        parser.add_argument(
            "--database_path",
            type=str,
            required=True,
            help="The path to the Retriever database",
        )
        return parser

    def __init__(self, args: Namespace):
        self.timeout = args.timeout
        self.log_interval = args.log_interval

        # set cache for retrieve
        if args.cache_size > 0:
            if not os.path.exists(args.database_path):
                os.makedirs(args.database_path)
            self._cache = PersistentLRUCache(
                persistant_path=os.path.join(args.database_path, "cache"),
                maxsize=args.cache_size,
            )
        else:
            self._cache = None
        return

    def search(
        self,
        query: list[str] | str,
        top_k: int = 10,
        retry_times: int = 3,
        delay: float = 0.1,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        if isinstance(query, str):
            query = [query]

        # prepare search method
        if retry_times > 1:
            search_method = retry(stop=stop_after_attempt(retry_times))(
                self.search_item
            )
        else:
            search_method = self.search_item

        # search
        results = []
        for n, q in enumerate(query):
            if n % self.log_interval == 0:
                logger.info(f"Searching for {n} / {len(query)}")
            results.append(search_method(q, top_k, delay=delay, **search_kwargs))
        return results

    def search_item(
        self,
        query: str,
        top_k: int = 10,
        delay: float = 0.1,
        disable_cache: bool = False,
        **search_kwargs,
    ) -> dict[str, str | list]:
        if (self._cache is None) or disable_cache:
            time.sleep(delay)  # delay only when cache is not available
            return self._search_item(query, top_k, **search_kwargs)

        # search from cache
        cache_key = hashkey(query=query, top_k=top_k, **search_kwargs)
        result = self._cache.get(cache_key)
        # search from database
        if result is None:
            time.sleep(delay)  # delay only when cache is not available
            result = self._search_item(query, top_k, **search_kwargs)
            self._cache[cache_key] = result
        return result

    @abstractmethod
    def _search_item(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> dict[str, str | list]:
        """Search queries using local retriever.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            dict[str, str | list]: The retrieved documents:
                {
                    "query": str,
                    "urls": list[str],
                    "titles": list[str],
                    "texts": list[str],
                }
        """
        return


class BingRetriever(WebRetriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--subscription_key",
            required=True,
            type=str,
            help="The api key for the Bing search engine",
        )
        parser.add_argument(
            "--endpoint",
            type=str,
            default="https://api.bing.microsoft.com",
            help="The endpoint for the Bing search engine",
        )
        parser = WebRetriever.add_args(parser)
        return parser

    def __init__(self, args: Namespace):
        super().__init__(args)
        self.endpoint = args.endpoint + "/v7.0/search"
        self.headers = {"Ocp-Apim-Subscription-Key": args.subscription_key}
        return

    def _search_item(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> dict[str, list]:
        params = {"q": query, "mkt": "en-US", "count": top_k}
        params.update(search_kwargs)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        results = response.json()
        results = {
            "query": query,
            "urls": [i["url"] for i in results["webPages"]["value"]],
            "texts": [i["snippet"] for i in results["webPages"]["value"]],
        }
        return results


class DuckDuckGoRetriever(WebRetriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--proxy",
            type=str,
            default=None,
            help="The proxy for the DuckDuckGo search engine",
        )
        parser = WebRetriever.add_args(parser)
        return parser

    def __init__(self, args: Namespace):
        super().__init__(args)

        from duckduckgo_search import DDGS

        self.ddgs = DDGS(proxy=args.proxy)
        return

    def _search_item(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> dict[str, list]:
        result = self.ddgs.text(query, max_results=top_k, **search_kwargs)
        result = {
            "query": query,
            "texts": [i["body"] for i in result],
            "urls": [i["href"] for i in result],
            "titles": [i["title"] for i in result],
        }
        return result
