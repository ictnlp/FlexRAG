import json
import time
from abc import abstractmethod

from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from flexrag.utils import LOGGER_MANAGER, TIME_METER, SimpleProgressLogger, configure
from flexrag.utils.configure import extract_config
from flexrag.utils.dataclasses import RetrievedContext

from ..retriever_base import (
    RETRIEVERS,
    RetrieverBase,
    RetrieverBaseConfig,
    batched_cache,
)
from .web_reader import WEB_READERS, WebReaderConfig
from .web_seeker import SEARCH_ENGINES, SearchEngineConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.web_retriever")


def _save_error_state(retry_state: RetryCallState) -> Exception:
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("web_retriever_error_state.json", "w", encoding="utf-8") as f:
        json.dump(args, f)
    raise retry_state.outcome.exception()


@configure
class WebRetrieverBaseConfig(RetrieverBaseConfig):
    """The configuration for the ``WebRetrieverBase``.

    :param retry_times: The number of times to retry. Default is 3.
    :type retry_times: int
    :param retry_delay: The delay between retries. Default is 0.5.
    :type retry_delay: float
    """

    retry_times: int = 3
    retry_delay: float = 0.5


class WebRetrieverBase(RetrieverBase):
    """The base class for the ``WebRetriever``.

    The WebRetriever is used to retrieve relevant information from the web.
    The subclasses should implement the ``search_item`` method.
    """

    cfg: WebRetrieverBaseConfig

    @TIME_METER("web_retriever", "search")
    @batched_cache
    def search(
        self,
        query: list[str] | str,
        delay: float = 0.1,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """As most web apis do not provide batch interface,
        we search the queries one by one using the ``search_item`` method."""
        if isinstance(query, str):
            query = [query]

        # prepare search method
        retry_times = search_kwargs.get("retry_times", self.cfg.retry_times)
        retry_delay = search_kwargs.get("retry_delay", self.cfg.retry_delay)
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
        p_logger = SimpleProgressLogger(logger, len(query), self.cfg.log_interval)
        top_k = search_kwargs.get("top_k", self.cfg.top_k)
        for q in query:
            time.sleep(delay)
            p_logger.update(1, "Searching")
            results.append(search_func(q, top_k, **search_kwargs))
        return results

    @abstractmethod
    def search_item(
        self,
        query: str,
        top_k: int,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        """Search the query from the web.

        :param query: The query to search.
        :type query: str
        :param top_k: The number of documents to return.
        :type top_k: int
        :return: The retrieved contexts.
        :rtype: list[RetrievedContext]
        """
        return


@configure
class SimpleWebRetrieverConfig(
    WebRetrieverBaseConfig,
    WebReaderConfig,
    SearchEngineConfig,
):
    """The configuration for the ``SimpleWebRetriever``."""


@RETRIEVERS("simple_web", config_class=SimpleWebRetrieverConfig)
class SimpleWebRetriever(WebRetrieverBase):
    """SimpleWebRetriever seeks most relevant web pages using existing search engine and reads the content using the WebReader."""

    def __init__(self, cfg: SimpleWebRetrieverConfig):
        super().__init__(cfg)
        # load the web page reader
        self.cfg = extract_config(cfg, SimpleWebRetrieverConfig)
        self.reader = WEB_READERS.load(cfg)
        assert self.reader is not None, "WebReader is not set."

        # load the search engine
        self.search_engine = SEARCH_ENGINES.load(cfg)
        return

    def search_item(
        self, query: str, top_k: int = 10, **search_kwargs
    ) -> RetrievedContext:
        resources = self.search_engine.seek(query, top_k=top_k, **search_kwargs)
        contexts = self.reader.read(resources)
        return contexts

    @property
    def fields(self):
        return self.reader.fields
