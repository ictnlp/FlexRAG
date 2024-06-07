import time
from abc import abstractmethod
from argparse import ArgumentParser, Namespace

import requests
from tenacity import retry, stop_after_attempt

from .retriever_base import Retriever


class WebRetriever(Retriever):
    @staticmethod
    def add_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument(
            "--delay",
            type=float,
            default=0.1,
            help="The delay between each request",
        )
        parser.add_argument(
            "--timeout",
            type=float,
            default=1.0,
            help="The timeout for each request",
        )
        return parser

    def __init__(self, args: Namespace):
        self.delay = args.delay
        self.timeout = args.timeout
        return

    def search(
        self,
        query: list[str],
        top_k: int = 10,
        retry_time: int = 3,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        results = []

        def delay_search(*args, **kwargs):
            time.sleep(self.delay)
            return self._search(*args, **kwargs)

        if retry_time > 1:
            search_method = retry(stop=stop_after_attempt(retry_time))(delay_search)
        else:
            search_method = delay_search
        for q in query:
            results.append(search_method(q, top_k, **search_kwargs))
        return super().search(query, top_k, **search_kwargs)

    @abstractmethod
    def _search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> dict[str, str | list]:
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

    def _search(
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

    def _search(
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
