import logging
import time
from abc import abstractmethod
from dataclasses import dataclass
from omegaconf import MISSING
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt

from .retriever_base import Retriever, RetrieverConfig


logger = logging.getLogger(__name__)


@dataclass
class WebRetrieverConfig(RetrieverConfig):
    timeout: float = 1.0


@dataclass
class BingRetrieverConfig(WebRetrieverConfig):
    subscription_key: str = MISSING
    endpoint: str = "https://api.bing.microsoft.com"


@dataclass
class DuckDuckGoRetrieverConfig(WebRetrieverConfig):
    proxy: Optional[str] = None


class WebRetriever(Retriever):
    name = "web"

    def __init__(self, cfg: WebRetrieverConfig):
        super().__init__(cfg)
        self.timeout = cfg.timeout
        return

    def _search(
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
            time.sleep(delay)
            if n % self.log_interval == 0:
                logger.info(f"Searching for {n} / {len(query)}")
            results.append(search_method(q, top_k, **search_kwargs))
        return results

    @abstractmethod
    def search_item(
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
    def __init__(self, cfg: BingRetrieverConfig):
        super().__init__(cfg)
        self.endpoint = cfg.endpoint + "/v7.0/search"
        self.headers = {"Ocp-Apim-Subscription-Key": cfg.subscription_key}
        return

    def search_item(
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
    def __init__(self, cfg: DuckDuckGoRetrieverConfig):
        super().__init__(cfg)

        from duckduckgo_search import DDGS

        self.ddgs = DDGS(proxy=cfg.proxy)
        return

    def search_item(
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
