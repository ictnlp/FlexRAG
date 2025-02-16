import time
from dataclasses import dataclass
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import LOGGER_MANAGER, SimpleProgressLogger

from ..retriever_base import RETRIEVERS, RetrieverBase, RetrieverBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.retrievers.web_retriever")


@dataclass
class WikipediaRetrieverConfig(RetrieverBaseConfig):
    """The configuration for the ``WikipediaRetriever``.

    :param search_url: The search URL for Wikipedia.
        Default is "https://en.wikipedia.org/w/index.php?search=".
    :type search_url: str
    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    """

    search_url: str = "https://en.wikipedia.org/w/index.php?search="
    proxy: Optional[str] = None


@RETRIEVERS("wikipedia", config_class=WikipediaRetrieverConfig)
class WikipediaRetriever(RetrieverBase):
    """WikipediaRetriever retrieves information from Wikipedia directly.
    Adapted from https://github.com/ysymyth/ReAct"""

    name = "wikipedia"

    def __init__(self, cfg: WikipediaRetrieverConfig):
        super().__init__(cfg)
        # set basic configs
        self.search_url = cfg.search_url
        self.client = httpx.Client(proxy=cfg.proxy)
        return

    def search(
        self,
        query: list[str] | str,
        delay: float = 0.1,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        if isinstance(query, str):
            query = [query]

        # search & parse
        results = []
        p_logger = SimpleProgressLogger(logger, len(query), self.log_interval)
        for q in query:
            time.sleep(delay)
            p_logger.update(1, "Searching")
            results.append([self.search_item(q, **search_kwargs)])
        return results

    def search_item(self, query: str, **kwargs) -> RetrievedContext:
        search_url = self.search_url + query.replace(" ", "+")
        response_text = self.client.get(search_url).text

        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            similar_entities = [
                self._clear_str(div.get_text().strip()) for div in result_divs
            ]
            page_content = None
            summary = None
        else:
            similar_entities = []
            page = [
                p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
            ]
            if any("may refer to:" in p for p in page):
                return self.search_item("[" + query + "]")
            else:  # concatenate all paragraphs
                page_content = ""
                for p in page:
                    if len(p.split(" ")) > 2:
                        page_content += self._clear_str(p)
                        if not p.endswith("\n"):
                            page_content += "\n"
                summary = self._get_summary(page_content)
        return RetrievedContext(
            retriever=self.name,
            query=query,
            data={
                "raw_page": response_text,
                "page_content": page_content,
                "summary": summary,
                "similar_entities": similar_entities,
            },
            source=search_url,
        )

    def _clear_str(self, text: str) -> str:
        return text.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

    def _get_summary(self, page: str) -> str:
        # find all paragraphs
        paragraphs = page.split("\n")
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        # find all sentence
        sentences = []
        for p in paragraphs:
            sentences += p.split(". ")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        summary = " ".join(sentences[:5])
        return summary.replace("\\n", "")

    @property
    def fields(self):
        return ["raw_page", "page_content", "summary", "similar_entities"]
