from abc import ABC, abstractmethod
from dataclasses import dataclass, field, make_dataclass
from typing import Optional

import requests
from omegaconf import MISSING

from kylin.models import GenerationConfig, GeneratorConfig, load_generator
from kylin.prompt import ChatPrompt, ChatTurn
from kylin.utils import Choices, Register

from ..retriever_base import RetrievedContext
from .web_downloader import WebDownloaderConfig, load_web_downloader


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
        return


WEB_READERS = Register("web_reader")


@dataclass
class JinaReaderLMConfig(GeneratorConfig, WebDownloaderConfig, GenerationConfig): ...


@WEB_READERS("jina_readerlm", config_class=JinaReaderLMConfig)
class JinaReaderLM(WebReader):
    def __init__(self, cfg: JinaReaderLMConfig):
        self.reader = load_generator(cfg)
        self.downloader = load_web_downloader(cfg)
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


web_reader_fields = [
    (
        "web_reader_type",
        Choices(WEB_READERS.names),
        "snippet",
    )
]
web_reader_fields += [
    (
        f"{WEB_READERS[name]['short_names'][0]}_config",
        WEB_READERS[name]["config_class"],
        field(default_factory=WEB_READERS[name]["config_class"]),
    )
    for name in WEB_READERS.mainnames
    if WEB_READERS[name]["config_class"] is not None
]
WebReaderConfig = make_dataclass("WebReaderConfig", web_reader_fields)


def load_web_reader(config: WebReaderConfig) -> WebReader:  # type: ignore
    config_name = f"{WEB_READERS[config.web_reader_type]['short_names'][0]}_config"
    sub_config = getattr(config, config_name, None)
    if sub_config is not None:
        return WEB_READERS[config.web_reader_type]["item"](sub_config)
    return WEB_READERS[config.web_reader_type]["item"]()
