import asyncio
import os
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from flexrag.utils import TIME_METER, Choices

from .model_base import ENCODERS, EncoderBase


@dataclass
class CohereEncoderConfig:
    """Configuration for CohereEncoder.

    :param model: The model to use. Default is "embed-multilingual-v3.0".
    :type model: str
    :param input_type: Specifies the type of input passed to the model. Required for embedding models v3 and higher. Default is "search_document". Available options are "search_document", "search_query", "classification", "clustering", "image".
    :type input_type: str
    :param base_url: The base URL of the API. Default is None.
    :type base_url: Optional[str]
    :param api_key: The API key to use. Default is os.environ.get("COHERE_API_KEY", MISSING).
    :type api_key: str
    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    """

    model: str = "embed-multilingual-v3.0"
    input_type: Choices(  # type: ignore
        [
            "search_document",
            "search_query",
            "classification",
            "clustering",
            "image",
        ]
    ) = "search_document"
    base_url: Optional[str] = None
    api_key: str = os.environ.get("COHERE_API_KEY", MISSING)
    proxy: Optional[str] = None


@ENCODERS("cohere", config_class=CohereEncoderConfig)
class CohereEncoder(EncoderBase):
    def __init__(self, cfg: CohereEncoderConfig):
        from cohere import ClientV2

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None
        self.client = ClientV2(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            httpx_client=httpx_client,
        )
        self.model = cfg.model
        self.input_type = cfg.input_type
        return

    @TIME_METER("cohere_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        r = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
            embedding_types=["float"],
        )
        embeddings = r.embeddings.float
        return np.array(embeddings)

    @TIME_METER("cohere_encode")
    async def async_encode(self, texts: list[str]):
        task = asyncio.create_task(
            asyncio.to_thread(
                self.client.embed,
                texts=texts,
                model=self.model,
                input_type=self.input_type,
                embedding_types=["float"],
            )
        )
        embeddings = (await task).embeddings.float
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        return self._data_template["dimension"]
