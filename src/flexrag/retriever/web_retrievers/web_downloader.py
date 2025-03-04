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

from flexrag.utils import Choices, Register
from .utils import WebResource


@dataclass
class WebDownloaderBaseConfig:
    """The configuration for the ``WebDownloaderBase``.

    :param allow_parallel: Whether to allow parallel downloading. Default is True.
    :type allow_parallel: bool
    """

    allow_parallel: bool = True


class WebDownloaderBase(ABC):
    """The base class for the ``WebDownloader``."""

    def __init__(self, cfg: WebDownloaderBaseConfig) -> None:
        self.allow_parallel = cfg.allow_parallel
        return

    def download(self, resources: WebResource | list[WebResource]) -> list[WebResource]:
        """Download the web resources.

        :param resources: The resources to download.
        :type resources: WebResource | list[WebResource]
        :return: The downloaded web resources.
        :rtype: list[WebResource]
        """
        if not isinstance(resources, list):
            resources = [resources]
        if self.allow_parallel:
            with ThreadPoolExecutor() as executor:
                results = list(executor.map(self.download_item, resources))
        else:
            results = [self.download_item(url) for url in resources]
        return results

    async def async_download(self, resources: WebResource | list[WebResource]) -> Any:
        """Download the web resources asynchronously."""
        if isinstance(resources, str):
            resources = [resources]
        results = await asyncio.gather(
            *[
                asyncio.to_thread(partial(self.download_item, url=url))
                for url in resources
            ]
        )
        return results

    @abstractmethod
    def download_item(self, resource: WebResource) -> Any:
        """Download the resource.

        :param resource: The web resource to download.
        :type resource: WebResource
        :return: The downloaded web resource.
        :rtype: WebResource
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
    :param headers: The headers to use. Default is None.
    :type headers: Optional[dict]
    """

    proxy: Optional[str] = None
    timeout: float = 3.0
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
        return

    def download_item(self, resource: WebResource) -> str:
        response = self.client.get(resource.url)
        response.raise_for_status()
        resource.data = response.text
        return resource


@dataclass
class PlaywrightWebDownloaderConfig(WebDownloaderBaseConfig):
    """The configuration for the ``PlaywrightWebDownloader``.

    :param headless: Whether to run the browser in headless mode. Default is True.
    :type headless: bool
    :param device: The device to emulate. Default is `Desktop Chrome`.
    :type device: str
    :param page_width: The width of the emulate device. Default is None.
    :type page_width: Optional[int]
    :param page_height: The height of the emulate device. Default is None.
    :type page_height: Optional[int]
    :param return_screenshot: Whether to return the screenshot. Default is False.
    :type return_screenshot: bool
    """

    headless: bool = True
    browser: Choices(["chromium", "firefox", "webkit", "msedge"]) = "chromium"  # type: ignore
    device: str = "Desktop Chrome"
    page_width: Optional[int] = None
    page_height: Optional[int] = None
    return_screenshot: bool = False


@WEB_DOWNLOADERS("playwright", config_class=PlaywrightWebDownloaderConfig)
class PlaywrightWebDownloader(WebDownloaderBase):
    """Download the web resources using playwright."""

    def __init__(self, cfg: PlaywrightWebDownloaderConfig) -> None:
        super().__init__(cfg)
        # load the playwright
        try:
            from playwright.async_api import async_playwright
            from playwright.sync_api import sync_playwright

            self.async_playwright = async_playwright
            self.sync_playwright = sync_playwright
        except ImportError:
            raise ImportError(
                "Please install playwright using `pip install pytest-playwright`."
                "Then, execute `playwright install`."
            )

        # set the arguments
        self.headless = cfg.headless
        self.browser = cfg.browser
        self.device = cfg.device
        self.page_width = cfg.page_width
        self.page_height = cfg.page_height
        self.return_screenshot = cfg.return_screenshot
        return

    def download(self, resources: WebResource | list[WebResource]) -> WebResource:
        if not isinstance(resources, list):
            resources = [resources]
        with self.sync_playwright() as p:
            # launch the browser
            match self.browser:
                case "chromium":
                    browser = p.chromium.launch(headless=self.headless)
                case "firefox":
                    browser = p.firefox.launch(headless=self.headless)
                case "webkit":
                    browser = p.webkit.launch(headless=self.headless)
                case "msedge":
                    browser = p.chromium.launch(headless=self.headless)
                case _:
                    raise ValueError(f"Browser {self.browser} is not supported.")

            # set the browser context
            ctx_param = p.devices[self.device]
            if self.page_height is not None:
                ctx_param["viewport"]["height"] = self.page_height
            if self.page_width is not None:
                ctx_param["viewport"]["width"] = self.page_width
            context = browser.new_context(**ctx_param)

            # download the resources
            if not self.allow_parallel:
                page = context.new_page()
                for r in resources:
                    page.goto(r.url)
                    if self.return_screenshot:
                        img_bytes = page.screenshot(full_page=True)
                        r.data = Image.open(io.BytesIO(img_bytes))
                    else:
                        r.data = page.content()
                page.close()
            else:

                def get_content(r: WebResource):
                    page = context.new_page()
                    page.goto(r.url)
                    if self.return_screenshot:
                        img_bytes = page.screenshot(full_page=True)
                        r.data = Image.open(io.BytesIO(img_bytes))
                    else:
                        r.data = page.content()
                    page.close()
                    return r

                with ThreadPoolExecutor() as executor:
                    resources = list(executor.map(get_content, resources))

            # close the browser
            browser.close()
        return resources

    async def async_download(self, resources):
        if not isinstance(resources, list):
            resources = [resources]
        async with self.async_playwright() as p:
            # launch the browser
            match self.browser:
                case "chromium":
                    browser = await p.chromium.launch(headless=self.headless)
                case "firefox":
                    browser = await p.firefox.launch(headless=self.headless)
                case "webkit":
                    browser = await p.webkit.launch(headless=self.headless)
                case "msedge":
                    browser = await p.chromium.launch(headless=self.headless)
                case _:
                    raise ValueError(f"Browser {self.browser} is not supported.")

            # set the browser context
            ctx_param = p.devices[self.device]
            if self.page_height is not None:
                ctx_param["viewport"]["height"] = self.page_height
            if self.page_width is not None:
                ctx_param["viewport"]["width"] = self.page_width
            context = await browser.new_context(**ctx_param)

            # download the resources
            async def get_content(r: WebResource):
                page = await context.new_page()
                await page.goto(r.url)
                if self.return_screenshot:
                    img_bytes = await page.screenshot(full_page=True)
                    r.data = Image.open(io.BytesIO(img_bytes))
                else:
                    r.data = await page.content()
                await page.close()
                return r

            resources = await asyncio.gather(*[get_content(r) for r in resources])

            # close the browser
            await browser.close()
        return resources

    def download_item(self, resource: WebResource) -> WebResource:
        return self.download(resource)


WebDownloaderConfig = WEB_DOWNLOADERS.make_config(config_name="WebDownloaderConfig")
