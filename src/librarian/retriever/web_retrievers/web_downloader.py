import asyncio
import io
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

from httpx import Client
from PIL import Image
from PIL.ImageFile import ImageFile
from tenacity import retry, stop_after_attempt, wait_fixed

from librarian.utils import Choices, Register


@dataclass
class BaseWebDownloaderConfig:
    allow_parallel: bool = True


class WebDownloaderBase(ABC):
    """The helper class for downloading web pages."""

    def __init__(self, cfg: BaseWebDownloaderConfig) -> None:
        self.allow_parallel = cfg.allow_parallel
        return

    def download(self, urls: str | list[str]) -> Any:
        if isinstance(urls, str):
            urls = [urls]
        if self.allow_parallel:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.download_page, urls))
        else:
            results = [self.download_page(url) for url in urls]
        return results

    async def async_download(self, urls: str | list[str]) -> Any:
        if isinstance(urls, str):
            urls = [urls]
        results = await asyncio.gather(
            *[asyncio.to_thread(partial(self.download_page, url=url)) for url in urls]
        )
        return results

    @abstractmethod
    def download_page(self, url: str) -> Any:
        return


WEB_DOWNLOADERS = Register[WebDownloaderBase]("web_downloader")


@dataclass
class SimpleWebDownloaderConfig:
    proxy: Optional[str] = None
    timeout: float = 3.0
    max_retries: int = 3
    retry_delay: float = 0.5
    skip_bad_response: bool = True
    headers: Optional[dict] = None


@WEB_DOWNLOADERS("simple", config_class=SimpleWebDownloaderConfig)
class SimpleWebDownloader(WebDownloaderBase):
    """Download the html content using httpx."""

    def __init__(self, cfg: SimpleWebDownloaderConfig) -> None:
        # setting httpx client
        self.client = Client(
            headers=cfg.headers,
            proxies=cfg.proxy,
            timeout=cfg.timeout,
        )

        # setting retry parameters
        self.skip_bad_response = cfg.skip_bad_response
        self.max_retries = cfg.max_retries
        self.retry_delay = cfg.retry_delay
        return

    def download_page(self, url: str) -> str:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_fixed(self.retry_delay),
            retry_error_callback=lambda _: None if self.skip_bad_response else None,
        )
        def download_page(url):
            response = self.client.get(url)
            response.raise_for_status()
            return response.text

        return download_page(url)


@dataclass
class PuppeteerWebDownloaderConfig(BaseWebDownloaderConfig):
    return_format: Choices(["screenshot", "html"]) = "html"  # type: ignore
    headless: bool = True
    device: Optional[str] = None
    page_width: int = 1280
    page_height: int = 1024


@WEB_DOWNLOADERS("puppeteer", config_class=PuppeteerWebDownloaderConfig)
class PuppeteerWebDownloader(WebDownloaderBase):
    """Download the web content using puppeteer."""

    def __init__(self, cfg: PuppeteerWebDownloaderConfig) -> None:
        super().__init__(cfg)
        # prepare the browser
        from pyppeteer import launch

        try:
            self.event_loop = asyncio.get_running_loop()
            self.browser = asyncio.create_task(launch(headless=cfg.headless))
        except RuntimeError:
            self.event_loop = asyncio.get_event_loop()
            self.browser = self.event_loop.run_until_complete(
                launch(headless=cfg.headless)
            )

        # setting the arguments
        self.return_format = cfg.return_format
        self.device = cfg.device
        self.page_width = cfg.page_width
        self.page_height = cfg.page_height
        return

    def download_page(self, url: str) -> ImageFile | str:
        return self.event_loop.run_until_complete(self.async_download_page(url))

    async def async_download_page(self, url: str) -> ImageFile | str:
        if isinstance(self.browser, asyncio.Task):
            self.browser = await self.browser
        page = await self.browser.newPage()
        if self.device is not None:
            await page.emulate(self.device)
        if self.page_width is not None and self.page_height is not None:
            await page.setViewport(
                {"width": self.page_width, "height": self.page_height}
            )
        await page.goto(url)
        if self.return_format == "html":
            content = await page.content()
        elif self.return_format == "screenshot":
            content = await page.screenshot(
                type="png", fullPage=True, encoding="binary"
            )
            content = Image.open(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported return format: {self.return_format}")
        await page.close()
        return content

    async def async_download(self, urls: str | list[str]) -> list[str | ImageFile]:
        return await asyncio.gather(*[self.async_download_page(url) for url in urls])
