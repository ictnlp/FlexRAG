import os
from typing import Annotated, Optional

import httpx
import numpy as np
from numpy import ndarray

from flexrag.utils import TIME_METER, Choices, configure

from .model_base import ENCODERS, EncoderBase, EncoderBaseConfig


@configure
class JinaEncoderConfig(EncoderBaseConfig):
    """Configuration for JinaEncoder.

    :param model: The model to use. Default is "jina-embeddings-v3".
    :type model: str
    :param base_url: The base URL of the Jina embeddings API. Default is "https://api.jina.ai/v1/embeddings".
    :type base_url: str
    :param api_key: The API key for the Jina embeddings API.
        If not provided, it will use the environment variable `JINA_API_KEY`.
        Defaults to None.
    :type api_key: str
    :param embedding_size: The dimension of the embeddings. Default is 1024.
    :type embedding_size: int
    :param task: The task for the embeddings. Default is None.
        Available options are "retrieval.query", "retrieval.passage", "separation", "classification", and "text-matching".
    :type task: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    """

    model: str = "jina-embeddings-v3"
    base_url: str = "https://api.jina.ai/v1/embeddings"
    api_key: Optional[str] = None
    embedding_size: int = 1024
    task: Optional[
        Annotated[
            str,
            Choices(
                "retrieval.query",
                "retrieval.passage",
                "separation",
                "classification",
                "text-matching",
            ),
        ]
    ] = None
    proxy: Optional[str] = None


@ENCODERS("jina", config_class=JinaEncoderConfig)
class JinaEncoder(EncoderBase):
    def __init__(self, cfg: JinaEncoderConfig):
        super().__init__(cfg)
        api_key = cfg.api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for Jina embeddings is not provided. "
                "Please set it in the configuration or as an environment variable 'JINA_API_KEY'."
            )
        # prepare client
        self.client = httpx.Client(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            proxy=cfg.proxy,
            base_url=cfg.base_url,
            follow_redirects=True,
        )
        self.async_client = httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            proxy=cfg.proxy,
            base_url=cfg.base_url,
            follow_redirects=True,
        )
        # prepare template
        self.data_template = {
            "model": cfg.model,
            "task": cfg.task,
            "dimensions": cfg.embedding_size,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [],
        }
        return

    @TIME_METER("encoder.jina_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        data = self.data_template.copy()
        data["input"] = texts
        response = self.client.post("", json=data)
        response.raise_for_status()
        embeddings = [i["embedding"] for i in response.json()["data"]]
        return np.array(embeddings)[:, : self.embedding_size]

    @TIME_METER("encoder.jina_encode")
    async def async_encode(self, texts: list[str]) -> ndarray:
        data = self.data_template.copy()
        data["input"] = texts
        response = await self.async_client.post("", json=data)
        embeddings = [i["embedding"] for i in (await response.json())["data"]]
        return np.array(embeddings)[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int:
        return self.data_template["dimensions"]
