import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import httpx
from omegaconf import MISSING

from flexrag.common_dataclass import RetrievedContext
from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import Register

from .web_downloader import (
    WEB_DOWNLOADERS,
    PuppeteerWebDownloader,
    PuppeteerWebDownloaderConfig,
    WebDownloaderConfig,
)


@dataclass
class WebRetrievedContext:
    """The context retrieved from the web.

    :param engine: The search engine used to retrieve the context.
    :type engine: str
    :param query: The query used to retrieve the context.
    :type query: str
    :param url: The URL of the context.
    :type url: str
    :param title: The title of the context. Default is None.
    :type title: Optional[str]
    :param snippet: The snippet of the context. Default is None.
    :type snippet: Optional[str]
    :param raw_content: The raw content of the context. Default is None.
    :type raw_content: Optional[dict]
    """

    engine: str = MISSING
    query: str = MISSING
    url: str = MISSING
    title: Optional[str] = None
    snippet: Optional[str] = None
    raw_content: Optional[dict] = None


class WebReaderBase(ABC):
    """The base class for the web readers."""

    @abstractmethod
    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        """
        Parse the retrieved contexts into LLM readable format.

        :param retrieved_contexts: Contexts retrieved by the WebRetriever.
        :type retrieved_contexts: list[WebRetrievedContext]
        :return: Contexts that can be fed into the LLM.
        :rtype: list[RetrievedContext]
        """
        return

    @property
    @abstractmethod
    def fields(self) -> list[str]:
        """The fields that the reader will return."""
        return


WEB_READERS = Register[WebReaderBase]("web_reader")


@dataclass
class JinaReaderLMConfig(GeneratorConfig, WebDownloaderConfig, GenerationConfig):
    """The configuration for the ``JinaReaderLM``.

    :param use_v2_prompt: Whether to use the jinaai/ReaderLM-v2 prompt. Default is False.
    :type use_v2_prompt: bool
    :param pre_clean_html: Whether to pre-clean the HTML content. Default is False.
    :type pre_clean_html: bool
    :param clean_svg: Whether to clean the SVG content. Default is False.
    :type clean_svg: bool
    :param clean_base64: Whether to clean the base64 images. Default is False.
    :type clean_base64: bool
    """

    use_v2_prompt: bool = False
    pre_clean_html: bool = False
    clean_svg: bool = False
    clean_base64: bool = False


@WEB_READERS("jina_readerlm", config_class=JinaReaderLMConfig)
class JinaReaderLM(WebReaderBase):
    """The JinaReaderLM reads the web pages using the Jina ReaderLM model."""

    # Patterns
    SCRIPT_PATTERN = r"<[ ]*script.*?\/[ ]*script[ ]*>"
    STYLE_PATTERN = r"<[ ]*style.*?\/[ ]*style[ ]*>"
    META_PATTERN = r"<[ ]*meta.*?>"
    COMMENT_PATTERN = r"<[ ]*!--.*?--[ ]*>"
    LINK_PATTERN = r"<[ ]*link.*?>"
    BASE64_IMG_PATTERN = r'<img[^>]+src="data:image/[^;]+;base64,[^"]+"[^>]*>'
    SVG_PATTERN = r"(<svg[^>]*>)(.*?)(<\/svg>)"

    def __init__(self, cfg: JinaReaderLMConfig):
        self.reader = GENERATORS.load(cfg)
        self.downloader = WEB_DOWNLOADERS.load(cfg)
        self.cfg = cfg
        if self.cfg.use_v2_prompt:
            self.template = (
                "Extract the main content from the given HTML and convert it to Markdown format."
                "\n```html\n{text}\n```"
            )
        else:
            self.template = "{text}"
        return

    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        urls = [rc.url for rc in retrieved_contexts]
        web_pages_ = self.downloader.download(urls)

        # Pre-clean the HTML content
        web_pages = []
        if self.cfg.pre_clean_html:
            for page in web_pages_:
                if page is not None:
                    web_pages.append(
                        JinaReaderLM.clean_html(
                            html=page,
                            clean_svg=self.cfg.clean_svg,
                            clean_base64=self.cfg.clean_base64,
                        )
                    )
                else:
                    web_pages.append(None)
        else:
            web_pages.append(page)

        # prepare prompts
        prompts = [
            ChatPrompt(
                history=[
                    ChatTurn(role="user", content=self.template.format(text=web_page))
                ]
            )
            for web_page in web_pages
            if web_page is not None
        ]

        # chat with the reader
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

    @staticmethod
    def replace_svg(html: str, new_content: str = "this is a placeholder") -> str:
        return re.sub(
            JinaReaderLM.SVG_PATTERN,
            lambda match: f"{match.group(1)}{new_content}{match.group(3)}",
            html,
            flags=re.DOTALL,
        )

    @staticmethod
    def replace_base64_images(html: str, new_image_src: str = "#") -> str:
        return re.sub(
            JinaReaderLM.BASE64_IMG_PATTERN,
            f'<img src="{new_image_src}"/>',
            html,
        )

    @staticmethod
    def clean_html(html: str, clean_svg: bool = False, clean_base64: bool = False):
        html = re.sub(
            JinaReaderLM.SCRIPT_PATTERN,
            "",
            html,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        html = re.sub(
            JinaReaderLM.STYLE_PATTERN,
            "",
            html,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        html = re.sub(
            JinaReaderLM.META_PATTERN,
            "",
            html,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        html = re.sub(
            JinaReaderLM.COMMENT_PATTERN,
            "",
            html,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        html = re.sub(
            JinaReaderLM.LINK_PATTERN,
            "",
            html,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )

        if clean_svg:
            html = JinaReaderLM.replace_svg(html)
        if clean_base64:
            html = JinaReaderLM.replace_base64_images(html)
        return html


@dataclass
class JinaReaderConfig:
    """The configuration for the ``JinaReader``.

    :param base_url: The base URL of the Jina Reader API. Default is "https://r.jina.ai".
    :type base_url: str
    :param api_key: The API key for the Jina Reader API. Default is os.environ.get("JINA_API_KEY", MISSING).
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    """

    base_url: str = "https://r.jina.ai"
    api_key: str = os.environ.get("JINA_API_KEY", MISSING)
    proxy: Optional[str] = None


@WEB_READERS("jina_reader", config_class=JinaReaderConfig)
class JinaReader(WebReaderBase):
    """The JinaReader reads the web pages using the Jina Reader API."""

    def __init__(self, cfg: JinaReaderConfig):
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={"Authorization": f"Bearer {cfg.api_key}"},
            proxy=cfg.proxy,
            follow_redirects=True,
        )
        return

    def read(
        self, retrieved_contexts: list[WebRetrievedContext]
    ) -> list[RetrievedContext]:
        responses = [self.client.get(f"/{rc.url}") for rc in retrieved_contexts]
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
class SnippetWebReader(WebReaderBase):
    """The SnippetWebReader will return the snippet of the web page directly.
    This is useful if the web pages are retrieved by the SearchEngines and the snippets are sufficient.
    """

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
class ScreenshotWebReaderConfig(PuppeteerWebDownloaderConfig):
    """The configuration for the ``ScreenshotWebReader``."""

    ...


@WEB_READERS("screenshot", config_class=ScreenshotWebReaderConfig)
class ScreenshotWebReader(WebReaderBase):
    """The ScreenshotWebReader reads the web pages by taking screenshots."""

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


WebReaderConfig = WEB_READERS.make_config(default="snippet")
