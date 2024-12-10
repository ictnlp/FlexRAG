from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import requests
from omegaconf import MISSING

from librarian.models import GenerationConfig, GENERATORS
from librarian.prompt import ChatPrompt, ChatTurn
from librarian.utils import Register

from ..retriever_base import RetrievedContext
from .web_downloader import (
    WEB_DOWNLOADERS,
    PuppeteerWebDownloaderConfig,
    PuppeteerWebDownloader,
)


@dataclass
class WebRetrievedContext:
    engine: str = MISSING
    query: str = MISSING
    url: str = MISSING
    title: Optional[str] = None
    snippet: Optional[str] = None
    raw_content: Optional[dict] = None


class WebReader(ABC):
    @abstractmethod
    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        """
        Parse the retrieved contexts into LLM readable format.

        Args:
            retrieved_contexts (list[WebRetrievedContext]): Contexts retrieved by the WebRetriever.

        Returns:
            list[RetrievedContext]: Contexts that can be fed into the LLM.
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields that the reader will return."""
        return


WEB_READERS = Register[WebReader]("web_reader")


GeneratorConfig = GENERATORS.make_config()
WebDownloaderConfig = WEB_DOWNLOADERS.make_config()


@dataclass
class JinaReaderLMConfig(GeneratorConfig, WebDownloaderConfig, GenerationConfig): ...


@WEB_READERS("jina_readerlm", config_class=JinaReaderLMConfig)
class JinaReaderLM(WebReader):
    def __init__(self, cfg: JinaReaderLMConfig):
        self.reader = GENERATORS.load(cfg)
        self.downloader = WEB_DOWNLOADERS.load(cfg)
        self.cfg = cfg
        return

    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        urls = [rc.url for rc in retrieved_contexts]
        web_pages = [self.downloader.download(url) for url in urls]
        prompts = [
            ChatPrompt(history=[ChatTurn(role="user", content=web_page)])
            for web_page in web_pages
            if web_page is not None
        ]
        texts = self.reader.chat(prompts, generation_config=self.cfg)
        texts = [t[0] for t in texts]
        contexts = []
        for p, ctx in zip(web_pages, retrieved_contexts):
            if p is None:
                continue
            contexts.append(
                RetrievedContext(
                    retriever=ctx.engine,
                    query=ctx.query,
                    data={"raw_content": p, "processed_content": texts.pop(0)},
                    source=ctx.url,
                )
            )
        return contexts

    @property
    def fields(self):
        return ["raw_content", "processed_content"]


@dataclass
class JinaReaderConfig:
    base_url: str = "https://r.jina.ai"
    api_key: str = MISSING


@WEB_READERS("jina_reader", config_class=JinaReaderConfig)
class JinaReader(WebReader):
    def __init__(self, cfg: JinaReaderConfig):
        self.base_url = cfg.base_url
        self.headers = {"Authorization": f"Bearer {cfg.api_key}"}
        return

    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        urls = [f"{self.base_url}/{rc.url}" for rc in retrieved_contexts]
        responses = [requests.get(url, headers=self.headers) for url in urls]
        contexts = []
        for rc, response in zip(retrieved_contexts, responses):
            if response.status_code == 200:
                contexts.append(
                    RetrievedContext(
                        retriever=rc.engine,
                        query=rc.query,
                        data={"processed_content": response.text},
                        source=rc.url,
                    )
                )
        return contexts

    @property
    def fields(self):
        return ["processed_content"]


@WEB_READERS("snippet")
class SnippetWebReader(WebReader):
    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        return [
            RetrievedContext(
                retriever=rc.engine,
                query=rc.query,
                data={"snippet": rc.snippet},
                source=rc.url,
            )
            for rc in retrieved_contexts
            if rc.snippet is not None
        ]

    @property
    def fields(self):
        return ["snippet"]


@dataclass
class ScreenshotWebReaderConfig(PuppeteerWebDownloaderConfig): ...


@WEB_READERS("screenshot", config_class=ScreenshotWebReaderConfig)
class ScreenshotWebReader(WebReader):
    def __init__(self, cfg: ScreenshotWebReaderConfig):
        super().__init__()
        assert cfg.return_format == "screenshot"
        self.downloader = PuppeteerWebDownloader(cfg)
        return

    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        urls = [rc.url for rc in retrieved_contexts]
        screenshots = self.downloader.download(urls)
        return [
            RetrievedContext(
                retriever=rc.engine,
                query=rc.query,
                data={"screenshot": screenshot},
                source=rc.url,
            )
            for rc, screenshot in zip(retrieved_contexts, screenshots)
        ]

    @property
    def fields(self):
        return ["screenshot"]
