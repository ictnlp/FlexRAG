import os
from typing import Annotated, Optional

import httpx
import numpy as np
from numpy import ndarray

from flexrag.common import TIME_METER, Choices, configure

from .api_based_encoder import APIBasedEncoder, APIBasedEncoderConfig
from .encoder_base import ENCODERS


@configure
class JinaEncoderConfig(APIBasedEncoderConfig):
    """Configuration for JinaEncoder.

    :param model: The model to use. Default is "jina-embeddings-v3".
    :type model: str
    :param base_url: The base URL of the Jina embeddings API. Default is "https://api.jina.ai/v1/embeddings".
    :type base_url: str
    :param api_key: The API key for the Jina embeddings API.
        If not provided, it will use the environment variable `JINA_API_KEY`.
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
    api_key: str = os.environ.get("JINA_API_KEY", "EMPTY")
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
class JinaEncoder(APIBasedEncoder):
    async def _create_client(self, config: JinaEncoderConfig):
        self._data_template = {
            "model": config.model,
            "task": config.task,
            "dimensions": config.embedding_size,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [],
        }
        return httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            proxy=config.proxy,
            base_url=config.base_url,
            follow_redirects=True,
        )

    @TIME_METER("encoder.jina_encode")
    async def _async_encode_impl(self, client, texts: list[str]) -> ndarray:
        data = self._data_template.copy()
        data["input"] = texts
        response = await client.post("", json=data)
        response.raise_for_status()
        embeddings = [i["embedding"] for i in response.json()["data"]]
        return np.array(embeddings)[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int:
        return self._data_template["dimensions"]
