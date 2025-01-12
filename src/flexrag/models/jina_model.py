import asyncio
from dataclasses import dataclass
from typing import Optional

import requests
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from flexrag.utils import TIME_METER, Choices

from .model_base import EncoderBase, ENCODERS


@dataclass
class JinaEncoderConfig:
    """Configuration for JinaEncoder.

    :param model: The model to use. Default is "jina-embeddings-v3".
    :type model: str
    :param base_url: The base URL of the Jina embeddings API. Default is "https://api.jina.ai/v1/embeddings".
    :type base_url: str
    :param api_key: The API key for the Jina embeddings API.
    :type api_key: str
    :param dimensions: The dimension of the embeddings. Default is 1024.
    :type dimensions: int
    :param task: The task for the embeddings. Default is None. Available options are "retrieval.query", "retrieval.passage", "separation", "classification", and "text-matching".
    :type task: str
    """

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
    def _encode(self, texts: list[str]) -> ndarray:
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
