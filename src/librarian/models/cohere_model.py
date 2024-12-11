import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from librarian.utils import TIME_METER

from .model_base import (
    EncoderBase,
    EncoderBaseConfig,
    ENCODERS,
)


@dataclass
class CohereEncoderConfig(EncoderBaseConfig):
    model: str = "embed-multilingual-v3.0"
    input_type: str = "search_document"
    base_url: Optional[str] = None
    api_key: str = MISSING
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
    def encode(self, texts: list[str]) -> ndarray:
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
