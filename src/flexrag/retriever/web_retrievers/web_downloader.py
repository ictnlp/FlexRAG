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

from flexrag.utils import Choices, Register


@dataclass
class WebDownloaderBaseConfig:
    """The configuration for the ``WebDownloaderBase``.

    :param allow_parallel: Whether to allow parallel downloading. Default is True.
    :type allow_parallel: bool
    """

    allow_parallel: bool = True


class WebDownloaderBase(ABC):
    """The base class for the web downloaders."""

    def __init__(self, cfg: WebDownloaderBaseConfig) -> None:
        self.allow_parallel = cfg.allow_parallel
        return

    def download(self, urls: str | list[str]) -> list[Any]:
        """Download the web pages.

        :param urls: The urls to download.
        :type urls: str | list[str]
        :return: The downloaded web pages.
        :rtype: list[Any]
        """
        if isinstance(urls, str):
            urls = [urls]
        if self.allow_parallel:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.download_page, urls))
        else:
            results = [self.download_page(url) for url in urls]
        return results

    async def async_download(self, urls: str | list[str]) -> Any:
        """Download the web pages asynchronously."""
        if isinstance(urls, str):
            urls = [urls]
        results = await asyncio.gather(
            *[asyncio.to_thread(partial(self.download_page, url=url)) for url in urls]
        )
        return results

    @abstractmethod
    def download_page(self, url: str) -> Any:
        """Download the web page.

        :param url: The url to download.
        :type url: str
        :return: The downloaded web pages.
        :rtype: Any
        """
        return


WEB_DOWNLOADERS = Register[WebDownloaderBase]("web_downloader")


@dataclass
class SimpleWebDownloaderConfig(WebDownloaderBaseConfig):
    """The configuration for the ``SimpleWebDownloader``.

    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    :param timeout: The timeout for the requests. Default is 3.0.
    :type timeout: float
    :param max_retries: The maximum number of retries. Default is 3.
    :type max_retries: int
    :param retry_delay: The delay between retries. Default is 0.5.
    :type retry_delay: float
    :param skip_bad_response: Whether to skip bad responses. Default is True.
    :type skip_bad_response: bool
    :param headers: The headers to use. Default is None.
    :type headers: Optional[dict]
    """

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
        super().__init__(cfg)
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
class PuppeteerWebDownloaderConfig(WebDownloaderBaseConfig):
    """The configuration for the ``PuppeteerWebDownloader``.

    :param return_format: The return format.
        Available options are "screenshot" and "html". Default is "html".
    :type return_format: str
    :param headless: Whether to run the browser in headless mode. Default is True.
    :type headless: bool
    :param device: The device to emulate. Default is None.
    :type device: Optional[str]
    :param page_width: The width of the emulate device. Default is 1280.
    :type page_width: int
    :param page_height: The height of the emulate device. Default is 1024.
    :type page_height: int
    """

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


WebDownloaderConfig = WEB_DOWNLOADERS.make_config()
