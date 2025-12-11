import os
from typing import Optional

import httpx
import numpy as np
from openai import AsyncAzureOpenAI, AsyncOpenAI, Omit

from flexrag.common import LOGGER_MANAGER, TIME_METER, configure

from .encoder_base import ENCODERS
from .remote_encoder_base import RemoteEncoderBase, RemoteEncoderBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.models.openai")


@configure
class OpenAIEncoderConfig(RemoteEncoderBaseConfig):
    """Configuration for OpenAI Encoder.

    :param is_azure: Whether the model is hosted on Azure. Default is False.
    :type is_azure: bool
    :param model_name: The name of the model to use.
    :type model_name: str
    :param base_url: The base URL of the OpenAI API. Default is None.
    :type base_url: Optional[str]
    :param api_key: The API key for OpenAI. Default is os.environ.get("OPENAI_API_KEY", "EMPTY").
    :type api_key: str
    :param api_version: The API version to use. Default is "2024-07-01-preview".
    :type api_version: str
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param proxy: The proxy to use for the HTTP client. Default is None.
    :type proxy: Optional[str]
    :param embedding_size: The size of the embedding vector.
        If None, it will be determined from the model.
        Default is None.
    :type embedding_size: Optional[int]
    """

    is_azure: bool = False
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_version: str = "2024-07-01-preview"
    verbose: bool = False
    proxy: Optional[str] = None
    embedding_size: Optional[int] = None


@ENCODERS("openai", config_class=OpenAIEncoderConfig)
class OpenAIEncoder(RemoteEncoderBase):
    async def _create_client(self, config: OpenAIEncoderConfig):
        """Create and return the async OpenAI client."""
        self._model_name = config.model_name
        self._dimension = config.embedding_size
        if config.proxy is not None:
            httpx_client = httpx.AsyncClient(proxies=config.proxy)
        else:
            httpx_client = None
        if config.is_azure:
            return AsyncAzureOpenAI(
                api_key=config.api_key,
                api_version=config.api_version,
                azure_endpoint=config.base_url,
                http_client=httpx_client,
            )
        else:
            return AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                http_client=httpx_client,
            )

    @TIME_METER("encoder.openai_encode")
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        dimension = self._dimension if self._dimension else Omit()
        r = await client.embeddings.create(
            model=self._model_name,
            input=texts,
            dimensions=dimension,
            encoding_format="float",
        )
        embeddings = [i.embedding for i in r.data]
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int | None:
        if self._dimension is None:
            match self._model_name:
                case "text-embedding-3-small":
                    return 1536
                case "text-embedding-3-large":
                    return 3072
                case "text-embedding-ada-002":
                    return 1536
                case _:
                    return None
        return self._dimension
