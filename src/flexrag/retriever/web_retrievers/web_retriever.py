import json
import os
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx
from omegaconf import MISSING
from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import LOGGER_MANAGER, TIME_METER, Choices, SimpleProgressLogger

from ..retriever_base import (
    RETRIEVERS,
    RetrieverBase,
    RetrieverBaseConfig,
    batched_cache,
)
from .web_reader import WEB_READERS, WebReaderConfig, WebRetrievedContext

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.web_retriever")


def _save_error_state(retry_state: RetryCallState) -> Exception:
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("web_retriever_error_state.json", "w", encoding="utf-8") as f:
        json.dump(args, f)
    raise retry_state.outcome.exception()


@dataclass
class WebRetrieverBaseConfig(RetrieverBaseConfig, WebReaderConfig):
    """The configuration for the ``WebRetrieverBase``.

    :param retry_times: The number of times to retry. Default is 3.
    :type retry_times: int
    :param retry_delay: The delay between retries. Default is 0.5.
    :type retry_delay: float
    """

    retry_times: int = 3
    retry_delay: float = 0.5


class WebRetrieverBase(RetrieverBase):
    """The base class for the web retrievers."""

    def __init__(self, cfg: WebRetrieverBaseConfig):
        super().__init__(cfg)
        # set retry parameters
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
        """Search the query using the search engine.

        :param query: The query to search.
        :type query: str
        :param top_k: The number of documents to return.
        :type top_k: int
        :return: The retrieved contexts.
        :rtype: list[WebRetrievedContext]
        """
        return

    @property
    def fields(self):
        return self.reader.fields


@dataclass
class BingRetrieverConfig(WebRetrieverBaseConfig):
    """The configuration for the ``BingRetriever``.

    :param subscription_key: The subscription key for the Bing Search API.
        Default is os.environ.get("BING_SEARCH_KEY", "EMPTY").
    :type subscription_key: str
    :param base_url: The base_url for the Bing Search API.
        Default is "https://api.bing.microsoft.com/v7.0/search".
    :type base_url: str
    :param timeout: The timeout for the requests. Default is 3.0.
    :type timeout: float
    """

    subscription_key: str = os.environ.get("BING_SEARCH_KEY", "EMPTY")
    base_url: str = "https://api.bing.microsoft.com/v7.0/search"
    timeout: float = 3.0


@RETRIEVERS("bing", config_class=BingRetrieverConfig)
class BingRetriever(WebRetrieverBase):
    """The BingRetriever retrieves the web pages using the Bing Search API."""

    name = "bing"

    def __init__(self, cfg: BingRetrieverConfig):
        super().__init__(cfg)
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Ocp-Apim-Subscription-Key": cfg.subscription_key},
            timeout=cfg.timeout,
            follow_redirects=True,
        )
        return

    def search_item(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebRetrievedContext]:
        params = {"q": query, "mkt": "en-US", "count": top_k}
        params.update(search_kwargs)
        response = self.client.get("", params=params)
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
class DuckDuckGoRetrieverConfig(WebRetrieverBaseConfig):
    """The configuration for the ``DuckDuckGoRetriever``.

    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    """

    proxy: Optional[str] = None


@RETRIEVERS("ddg", config_class=DuckDuckGoRetrieverConfig)
class DuckDuckGoRetriever(WebRetrieverBase):
    """The DuckDuckGoRetriever retrieves the web pages using the DuckDuckGo Search API."""

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
class GoogleRetrieverConfig(WebRetrieverBaseConfig):
    """The configuration for the ``GoogleRetriever``.

    :param subscription_key: The subscription key for the Google Search API.
        Default is os.environ.get("GOOGLE_SEARCH_KEY", "EMPTY").
    :type subscription_key: str
    :param search_engine_id: The search engine id for the Google Search API.
        Default is os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "EMPTY").
    :type search_engine_id: str
    :param endpoint: The endpoint for the Google Search API.
        Default is "https://customsearch.googleapis.com/customsearch/v1".
    :type endpoint: str
    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    :param timeout: The timeout for the requests. Default is 3.0.
    :type timeout: float
    """

    subscription_key: str = os.environ.get("GOOGLE_SEARCH_KEY", "EMPTY")
    search_engine_id: str = os.environ.get("GOOGLE_SEARCH_ENGINE_ID", "EMPTY")
    endpoint: str = "https://customsearch.googleapis.com/customsearch/v1"
    proxy: Optional[str] = None
    timeout: float = 3.0


@RETRIEVERS("google", config_class=GoogleRetrieverConfig)
class GoogleRetriever(WebRetrieverBase):
    """The GoogleRetriever retrieves the web pages using the `Google Custom Search` API."""

    name = "google"

    def __init__(self, cfg: GoogleRetrieverConfig):
        super().__init__(cfg)
        self.subscription_key = cfg.subscription_key
        self.engine_id = cfg.search_engine_id
        self.client = httpx.Client(
            base_url=cfg.endpoint,
            timeout=cfg.timeout,
            proxy=cfg.proxy,
            follow_redirects=True,
        )
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
        response = self.client.get("", params=params)
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
class SerpApiRetrieverConfig(WebRetrieverBaseConfig):
    """The configuration for the ``SerpApiRetriever``.

    :param api_key: The API key for the SerpApi.
        Default is os.environ.get("SERP_API_KEY", MISSING).
    :type api_key: str
    :param engine: The search engine to use. Default is "google".
        Available choices are "google", "bing", "baidu", "yandex", "yahoo", "google_scholar", "duckduckgo".
    :type engine: str
    :param country: The country to search. Default is "us".
    :type country: str
    :param language: The language to search. Default is "en".
    :type language: str
    """

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
    """The SerpApiRetriever retrieves the web pages using the `SerpApi <https://serpapi.com/>_`."""

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
