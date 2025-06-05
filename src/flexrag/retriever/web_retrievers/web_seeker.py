import os
from abc import ABC, abstractmethod
from typing import Annotated, Optional

import httpx

from flexrag.utils import Choices, Register, configure

from .utils import WebResource


class WebSeekerBase(ABC):
    """The base class for the WebSeeker.
    The WebSeeker is used to seek the web resources for a given query.
    The web resources could be sought by walking through a set of given web pages, by using a search engine, etc.

    The subclasses should implement the ``seek`` method.
    """

    @abstractmethod
    def seek(self, query: str, top_k: int = 10, **kwargs) -> list[WebResource]:
        """Seek the web resources.

        :param query: The query to seek.
        :type query: str
        :param top_k: The number of resources to seek. Default is 10.
        :type top_k: int
        :param kwargs: The additional keyword arguments.
        :return: The web resources.
        :rtype: list[WebResource]
        """
        raise NotImplementedError


WEB_SEEKERS = Register[WebSeekerBase]("web_seeker")
SEARCH_ENGINES = Register[WebSeekerBase]("search_engine")


@configure
class BingEngineConfig:
    """The configuration for the ``BingSeeker``.

    :param subscription_key: The subscription key for the Bing Search API.
        Default is os.environ.get("BING_SEARCH_KEY", "EMPTY").
    :type subscription_key: str
    :param base_url: The base_url for the Bing Search API.
        Default is "https://api.bing.microsoft.com/v7.0/search".
    :type base_url: str
    :param timeout: The timeout for the requests. Default is 3.0.
    :type timeout: float
    :param market: The market to use.
        see https://learn.microsoft.com/en-us/bing/search-apis/bing-web-search/reference/market-codes for more information).
        Default is en-US.
    :type market: str
    :param lang: The language to use. Default is "en".
    :type lang: str
    :param freshness: To get articles discovered by Bing during a specific timeframe,
        specify a date range in the form, YYYY-MM-DD..YYYY-MM-DD.
        For example, &freshness=2019-02-01..2019-05-30.
        Default is None.
    :type freshness: Optional[str]
    """

    subscription_key: str = os.environ.get("BING_SEARCH_KEY", "EMPTY")
    base_url: str = "https://api.bing.microsoft.com/v7.0/search"
    timeout: float = 3.0
    market: str = "en-US"
    lang: str = "en"
    freshness: Optional[str] = None


@SEARCH_ENGINES("bing", config_class=BingEngineConfig)
@WEB_SEEKERS("bing", config_class=BingEngineConfig)
class BingEngine(WebSeekerBase):
    """The BingSeeker retrieves the web pages using the Bing Search API."""

    def __init__(self, cfg: BingEngineConfig):
        super().__init__()
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Ocp-Apim-Subscription-Key": cfg.subscription_key},
            timeout=cfg.timeout,
            follow_redirects=True,
        )
        self.market = cfg.market
        self.freshness = cfg.freshness
        self.lang = cfg.lang
        return

    def seek(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebResource]:
        params = {"q": query, "mkt": self.market, "count": top_k, "setLang": self.lang}
        if self.freshness is not None:
            params["freshness"] = self.freshness
        params.update(search_kwargs)
        response = self.client.get("", params=params)
        response.raise_for_status()
        result = response.json()
        if "webPages" not in result:
            return []
        result = [
            WebResource(
                query=query,
                url=i["url"],
                metadata={
                    "engine": "Bing",
                    "snippet": i["snippet"],
                },
            )
            for i in result["webPages"]["value"]
        ]
        return result


@configure
class DuckDuckGoEngineConfig:
    """The configuration for the ``DuckDuckGoEngine``.

    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    """

    proxy: Optional[str] = None


@SEARCH_ENGINES("ddg", config_class=DuckDuckGoEngineConfig)
@WEB_SEEKERS("ddg", config_class=DuckDuckGoEngineConfig)
class DuckDuckGoEngine(WebSeekerBase):
    """The DuckDuckGoEngine retrieves the web pages using the DuckDuckGo Search API."""

    def __init__(self, cfg: DuckDuckGoEngineConfig):
        super().__init__()

        from duckduckgo_search import DDGS

        self.ddgs = DDGS(proxy=cfg.proxy)
        return

    def seek(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebResource]:
        result = self.ddgs.text(query, max_results=top_k, **search_kwargs)
        result = [
            WebResource(
                query=query,
                url=i["href"],
                metadata={
                    "engine": "DuckDuckGo",
                    "title": i["title"],
                    "snippet": i["body"],
                },
            )
            for i in result
        ]
        return result


@configure
class GoogleEngineConfig:
    """The configuration for the ``GoogleEngine``.

    :param subscription_key: The subscription key for the Google Search API.
        If not provided, it will use the environment variable `GOOGLE_SEARCH_KEY`.
        Defaults to None.
    :type subscription_key: str
    :param search_engine_id: The search engine id for the Google Search API.
        If not provided, it will use the environment variable `GOOGLE_SEARCH_ENGINE_ID`.
        Defaults to None.
    :type search_engine_id: str
    :param endpoint: The endpoint for the Google Search API.
        Default is "https://customsearch.googleapis.com/customsearch/v1".
    :type endpoint: str
    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    :param timeout: The timeout for the requests. Default is 3.0.
    :type timeout: float
    """

    subscription_key: Optional[str] = None
    search_engine_id: Optional[str] = None
    endpoint: str = "https://customsearch.googleapis.com/customsearch/v1"
    proxy: Optional[str] = None
    timeout: float = 3.0


@SEARCH_ENGINES("google", config_class=GoogleEngineConfig)
@WEB_SEEKERS("google", config_class=GoogleEngineConfig)
class GoogleEngine(WebSeekerBase):
    """The GoogleEngine retrieves the web pages using the `Google Custom Search` API."""

    name = "google"

    def __init__(self, cfg: GoogleEngineConfig):
        super().__init__()
        self.subscription_key = cfg.subscription_key or os.getenv("GOOGLE_SEARCH_KEY")
        if not self.subscription_key:
            raise ValueError(
                "Subscription key for Google Search API is not provided. "
                "Please set it in the configuration or as an environment variable 'GOOGLE_SEARCH_KEY'."
            )
        self.engine_id = cfg.search_engine_id or os.getenv("GOOGLE_SEARCH_ENGINE_ID")
        if not self.engine_id:
            raise ValueError(
                "Search engine ID for Google Search API is not provided. "
                "Please set it in the configuration or as an environment variable 'GOOGLE_SEARCH_ENGINE_ID'."
            )
        self.client = httpx.Client(
            base_url=cfg.endpoint,
            timeout=cfg.timeout,
            proxy=cfg.proxy,
            follow_redirects=True,
        )
        return

    def seek(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebResource]:
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
            WebResource(
                query=query,
                url=i["link"],
                metadata={
                    "engine": "Google",
                    "title": i["title"],
                    "snippet": i["snippet"],
                },
            )
            for i in result["items"]
        ]
        return result


@configure
class SerpApiConfig:
    """The configuration for the ``SerpApi``.

    :param api_key: The API key for the SerpApi.
        If not provided, it will use the environment variable `SERPAPI_API_KEY`.
        Defaults to None.
    :type api_key: str
    :param engine: The search engine to use. Default is "google".
        Available choices are "google", "bing", "baidu", "yandex", "yahoo", "google_scholar", "duckduckgo".
    :type engine: str
    :param country: The country to search. Default is "us".
    :type country: str
    :param language: The language to search. Default is "en".
    :type language: str
    """

    api_key: Optional[str] = None
    engine: Annotated[
        str,
        Choices(
            "google",
            "bing",
            "baidu",
            "yandex",
            "yahoo",
            "google_scholar",
            "duckduckgo",
        ),
    ] = "google"
    country: str = "us"
    language: str = "en"


@SEARCH_ENGINES("serpapi", config_class=SerpApiConfig)
@WEB_SEEKERS("serpapi", config_class=SerpApiConfig)
class SerpApi(WebSeekerBase):
    """The SerpApi retrieves the web pages using the `SerpApi <https://serpapi.com/>_`."""

    def __init__(self, cfg: SerpApiConfig):
        super().__init__()
        api_key = cfg.api_key or os.getenv("SERPAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for SerpApi is not provided. "
                "Please set it in the configuration or as an environment variable 'SERPAPI_API_KEY'."
            )
        try:
            import serpapi

            self.client = serpapi.Client(api_key=api_key)
        except ImportError:
            raise ImportError("Please install serpapi with `pip install serpapi`.")

        self.api_key = api_key
        self.engine = cfg.engine
        self.gl = cfg.country
        self.hl = cfg.language
        return

    def seek(
        self,
        query: str,
        top_k: int = 10,
        **search_kwargs,
    ) -> list[WebResource]:
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
            WebResource(
                query=query,
                url=r["link"],
                metadata={
                    "engine": "SerpAPI",
                    "title": r.get("title", None),
                    "snippet": r.get("snippet", None),
                },
            )
            for r in data["organic_results"]
        ]
        return contexts


WebSeekerConfig = WEB_SEEKERS.make_config(config_name="WebSeekerConfig")
SearchEngineConfig = SEARCH_ENGINES.make_config(config_name="SearchEngineConfig")
