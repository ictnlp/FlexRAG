import json
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests
from omegaconf import MISSING
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from librarian.utils import LOGGER_MANAGER, Choices, SimpleProgressLogger, TIME_METER

from ..retriever_base import (
    RETRIEVERS,
    RetrievedContext,
    RetrieverBase,
    RetrieverConfigBase,
    batched_cache,
)
from .web_reader import WEB_READERS, WebRetrievedContext

logger = LOGGER_MANAGER.get_logger("librarian.retrievers.web_retriever")


def _save_error_state(retry_state: RetryCallState) -> Exception:
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("web_retriever_error_state.json", "w") as f:
        json.dump(args, f)
    raise retry_state.outcome.exception()


WebReaderConfig = WEB_READERS.make_config(default="snippet")


@dataclass
class WebRetrieverConfig(RetrieverConfigBase, WebReaderConfig):
    timeout: float = 3.0
    retry_times: int = 3
    retry_delay: float = 0.5


class WebRetrieverBase(RetrieverBase):
    def __init__(self, cfg: WebRetrieverConfig):
        super().__init__(cfg)
        # set retry parameters
        self.timeout = cfg.timeout
        self.retry_times = cfg.retry_times
        self.retry_delay = cfg.retry_delay
        # load web reader
        self.reader = WEB_READERS.load(cfg)
        return

    @TIME_METER("web_retriever", "search")
    @batched_cache
    def search(
        self,
        query: list[str] | str,
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

        # search & parse
        results = []
        p_logger = SimpleProgressLogger(logger, len(query), self.log_interval)
        top_k = search_kwargs.get("top_k", self.top_k)
        for q in query:
            time.sleep(delay)
            p_logger.update(1, "Searching")
            results.append(self.reader.read(search_func(q, top_k, **search_kwargs)))
        return results

    @abstractmethod
    def search_item(
        self,
        query: str,
        top_k: int,
        **search_kwargs,
    ) -> list[WebRetrievedContext]:
        """Search queries using local retriever.

        Args:
            query (str): Query to search.
            top_k (int, optional): N documents to return.

        Returns:
            list[WebRetrievedContext]: k WebRetrievedContext.
        """
        return

    @property
    def fields(self):
        return self.reader.fields


@dataclass
class BingRetrieverConfig(WebRetrieverConfig):
    subscription_key: str = os.environ.get("BING_SEARCH_KEY", "EMPTY")
    endpoint: str = "https://api.bing.microsoft.com"


@RETRIEVERS("bing", config_class=BingRetrieverConfig)
class BingRetriever(WebRetrieverBase):
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
    ) -> list[WebRetrievedContext]:
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
            WebRetrievedContext(
                engine=self.name,
                query=query,
                url=i["url"],
                snippet=i["snippet"],
            )
            for i in result["webPages"]["value"]
        ]
        return result


@dataclass
class DuckDuckGoRetrieverConfig(WebRetrieverConfig):
    proxy: Optional[str] = None


@RETRIEVERS("ddg", config_class=DuckDuckGoRetrieverConfig)
class DuckDuckGoRetriever(WebRetrieverBase):
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
    ) -> list[WebRetrievedContext]:
        result = self.ddgs.text(query, max_results=top_k, **search_kwargs)
        result = [
            WebRetrievedContext(
                engine=self.name,
                query=query,
                url=i["href"],
                title=i["title"],
                snippet=i["body"],
            )
            for i in result
        ]
        return result


@dataclass
class GoogleRetrieverConfig(WebRetrieverConfig):
    subscription_key: str = os.environ.get("GOOGLE_SEARCH_KEY", "EMPTY")
    search_engine_id: str = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "EMPTY")
    endpoint: str = "https://customsearch.googleapis.com/customsearch/v1"
    proxy: Optional[str] = None


@RETRIEVERS("google", config_class=GoogleRetrieverConfig)
class GoogleRetriever(WebRetrieverBase):
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
    ) -> list[WebRetrievedContext]:
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
            WebRetrievedContext(
                engine=self.name,
                query=query,
                title=i["title"],
                url=i["link"],
                snippet=i["snippet"],
            )
            for i in result["items"]
        ]
        return result


@dataclass
class SerpApiRetrieverConfig(WebRetrieverConfig):
    api_key: str = os.environ.get("SERP_API_KEY", MISSING)
    engine: Choices(  # type: ignore
        [
            "google",
            "bing",
            "baidu",
            "yandex",
            "yahoo",
            "google_scholar",
            "duckduckgo",
        ]
    ) = "google"
    country: str = "us"
    language: str = "en"


@RETRIEVERS("serpapi", config_class=SerpApiRetrieverConfig)
class SerpApiRetriever(WebRetrieverBase):
    def __init__(self, cfg: SerpApiRetrieverConfig):
        super().__init__(cfg)
        try:
            import serpapi

            self.client = serpapi.Client(api_key=cfg.api_key)
        except ImportError:
            raise ImportError("Please install serpapi with `pip install serpapi`.")

        self.api_key = cfg.api_key
        self.engine = cfg.engine
        self.gl = cfg.country
        self.hl = cfg.language
        return

    def search_item(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebRetrievedContext]:
        search_params = {
            "q": query,
            "engine": self.engine,
            "api_key": self.api_key,
            "gl": self.gl,
            "hl": self.hl,
            "num": top_k,
        }
        search_params.update(search_kwargs)
        data = self.client.search(search_params)
        contexts = [
            WebRetrievedContext(
                engine=self.engine,
                query=query,
                url=r["link"],
                title=r.get("title", None),
                snippet=r.get("snippet", None),
            )
            for r in data["organic_results"]
        ]
        return contexts
