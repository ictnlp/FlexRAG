import json
import logging
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional
from uuid import NAMESPACE_OID, uuid5

import requests
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from kylin.utils import SimpleProgressLogger

from .retriever_base import WEB_RETRIEVERS, RetrievedContext, Retriever, RetrieverConfig

logger = logging.getLogger(__name__)


def _save_error_state(retry_state: RetryCallState) -> Exception:
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("web_retriever_error_state.json", "w") as f:
        json.dump(args, f)
    raise retry_state.outcome.exception()


@dataclass
class WebRetrieverConfig(RetrieverConfig):
    timeout: float = 3.0
    retry_times: int = 3
    retry_delay: float = 0.5


@dataclass
class BingRetrieverConfig(WebRetrieverConfig):
    subscription_key: str = os.environ.get("BING_SEARCH_KEY", "EMPTY")
    endpoint: str = "https://api.bing.microsoft.com"


@dataclass
class DuckDuckGoRetrieverConfig(WebRetrieverConfig):
    proxy: Optional[str] = None


@dataclass
class GoogleRetrieverConfig(WebRetrieverConfig):
    subscription_key: str = os.environ.get("GOOGLE_SEARCH_KEY", "EMPTY")
    search_engine_id: str = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "EMPTY")
    endpoint: str = "https://customsearch.googleapis.com/customsearch/v1"
    proxy: Optional[str] = None


class WebRetriever(Retriever):
    def __init__(self, cfg: WebRetrieverConfig):
        super().__init__(cfg)
        self.timeout = cfg.timeout
        self.retry_times = cfg.retry_times
        self.retry_delay = cfg.retry_delay
        return

    def _search(
        self,
        query: list[str] | str,
        top_k: int = 10,
        delay: float = 0.1,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        if isinstance(query, str):
            query = [query]

        # prepare search method
        retry_times = search_kwargs.get("retry_times", self.retry_times)
        retry_delay = search_kwargs.get("retry_delay", self.retry_delay)
        if retry_times > 1:
            search_func = retry(
                stop=stop_after_attempt(retry_times),
                wait=wait_fixed(retry_delay),
                retry_error_callback=_save_error_state,
            )(self.search_item)
        else:
            search_func = self.search_item

        # search
        results = []
        p_logger = SimpleProgressLogger(logger, len(query), self.log_interval)
        for q in query:
            time.sleep(delay)
            p_logger.update(1, "Searching")
            results.append(search_func(q, top_k, **search_kwargs))
        return results

    @abstractmethod
    def search_item(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        """Search queries using local retriever.

        Args:
            query (list[str]): Queries to search.
            top_k (int, optional): N documents to return. Defaults to 10.

        Returns:
            list[RetrievedContext]: k RetrievedContext
        """
        return


@WEB_RETRIEVERS("bing", config_class=BingRetrieverConfig)
class BingRetriever(WebRetriever):
    name = "bing"

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
    ) -> list[RetrievedContext]:
        params = {"q": query, "mkt": "en-US", "count": top_k}
        params.update(search_kwargs)
        response = requests.get(
            self.endpoint,
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        if "webPages" not in result:
            return []
        result = [
            RetrievedContext(
                retriever=self.name,
                query=query,
                source=i["url"],
                text=i["snippet"],
                full_text=i["snippet"],
                title=i["name"],
                chunk_id=i["id"],
            )
            for i in result["webPages"]["value"]
        ]
        return result

    @property
    def fingerprint(self) -> str:
        time_str = time.strftime("%Y-%m", time.gmtime())
        namespace_uuid = uuid5(NAMESPACE_OID, self.name)
        feature_str = self.name + self.endpoint + json.dumps(self.headers)
        return uuid5(namespace_uuid, time_str + feature_str).hex


@WEB_RETRIEVERS("ddg", config_class=DuckDuckGoRetrieverConfig)
class DuckDuckGoRetriever(WebRetriever):
    name = "ddg"

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
    ) -> list[RetrievedContext]:
        result = self.ddgs.text(query, max_results=top_k, **search_kwargs)
        result = [
            RetrievedContext(
                retriever=self.name,
                query=query,
                source=i["href"],
                text=i["body"],
                full_text=i["body"],
                title=i["title"],
            )
            for i in result
        ]
        return result

    @property
    def fingerprint(self) -> str:
        time_str = time.strftime("%Y-%m", time.gmtime())
        namespace_uuid = uuid5(NAMESPACE_OID, self.name)
        return uuid5(namespace_uuid, time_str).hex


@WEB_RETRIEVERS("google", config_class=GoogleRetrieverConfig)
class GoogleRetriever(WebRetriever):
    name = "google"

    def __init__(self, cfg: GoogleRetrieverConfig):
        super().__init__(cfg)
        self.endpoint = cfg.endpoint
        self.subscription_key = cfg.subscription_key
        self.engine_id = cfg.search_engine_id
        self.proxy = {
            "http": cfg.proxy,
            "https": cfg.proxy,
        }
        return

    def search_item(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        params = {
            "key": self.subscription_key,
            "cx": self.engine_id,
            "q": query,
            "num": top_k,
        }
        response = requests.get(
            self.endpoint,
            params=params,
            proxies=self.proxy,
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()
        result = [
            RetrievedContext(
                retriever=self.name,
                query=query,
                source=i["link"],
                text=i["snippet"],
                full_text=i["snippet"],
                title=i["title"],
            )
            for i in result["items"]
        ]
        return result

    @property
    def fingerprint(self) -> str:
        time_str = time.strftime("%Y-%m", time.gmtime())
        namespace_uuid = uuid5(NAMESPACE_OID, self.name)
        feature_str = self.engine_id + self.endpoint
        return uuid5(namespace_uuid, time_str + feature_str).hex
