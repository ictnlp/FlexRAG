import asyncio
from dataclasses import dataclass
from typing import Optional

import requests
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from librarian.utils import TIME_METER, Choices

from .model_base import (
    EncoderBase,
    EncoderBaseConfig,
    ENCODERS,
)


@dataclass
class JinaEncoderConfig(EncoderBaseConfig):
    model: str = "jina-embeddings-v3"
    base_url: str = "https://api.jina.ai/v1/embeddings"
    api_key: str = MISSING
    dimensions: int = 1024
    task: Optional[
        Choices(  # type: ignore
            [
                "retrieval.query",
                "retrieval.passage",
                "separation",
                "classification",
                "text-matching",
            ]
        )
    ] = None


@ENCODERS("jina", config_class=JinaEncoderConfig)
class JinaEncoder(EncoderBase):
    def __init__(self, cfg: JinaEncoderConfig):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        self.base_url = cfg.base_url
        self._data_template = {
            "model": cfg.model,
            "task": cfg.task,
            "dimensions": cfg.dimensions,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [],
        }
        return

    @TIME_METER("jina_encode")
    def encode(self, texts: list[str]) -> ndarray:
        data = self._data_template.copy()
        data["input"] = texts
        response = requests.post(self.base_url, headers=self.headers, json=data)
        response.raise_for_status()
        embeddings = [i["embedding"] for i in response.json()["data"]]
        return np.array(embeddings)

    @TIME_METER("jina_encode")
    async def async_encode(self, texts: list[str]) -> ndarray:
        data = self._data_template.copy()
        data["input"] = texts
        response = await asyncio.to_thread(
            requests.post, self.base_url, headers=self.headers, json=data
        )
        embeddings = [i["embedding"] for i in response.json()["data"]]
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        return self._data_template["dimension"]
