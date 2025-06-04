import asyncio
import os
from typing import Annotated, Optional

import httpx
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from flexrag.utils import TIME_METER, Choices, configure

from .model_base import ENCODERS, EncoderBase, EncoderBaseConfig


@configure
class CohereEncoderConfig(EncoderBaseConfig):
    """Configuration for CohereEncoder.

    :param model: The model to use. Default is "embed-v4.0".
    :type model: str
    :param input_type: Specifies the type of input passed to the model.
        Required for embedding models v3 and higher. Default is "search_document".
        Available options are "search_document", "search_query", "classification", "clustering", "image".
    :type input_type: str
    :param embedding_size: The size of the embedding. Default is "1536".
        Available options are "256", "512", "1024", "1536".
        This option is only used for embedding models v4 and newer.
    :type embedding_size: str
    :param base_url: The base URL of the API. Default is None.
    :type base_url: Optional[str]
    :param api_key: The API key to use. Default is os.environ.get("COHERE_API_KEY", MISSING).
    :type api_key: str
    :param proxy: The proxy to use. Default is None.
    :type proxy: Optional[str]
    """

    model: str = "embed-v4.0"
    input_type: Annotated[
        str,
        Choices(
            "search_document",
            "search_query",
            "classification",
            "clustering",
            "image",
        ),
    ] = "search_document"
    embedding_size: Annotated[str, Choices("256", "512", "1024", "1536")] = "1536"
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
        self._embedding_size = int(cfg.embedding_size)
        super().__init__(cfg)
        return

    @TIME_METER("cohere_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        embed_dim = self.embedding_size if self.model == "embed-v4.0" else None
        r = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
            embedding_types=["float"],
            output_dimension=embed_dim,
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
        match self.model:
            case "embed-multilingual-light-v3.0":
                return 384
            case "embed-multilingual-v3.0":
                return 1024
            case "embed-english-light-v3.0":
                return 384
            case "embed-english-v3.0":
                return 1024
            case "embed-v4.0":
                if self._embedding_size is not None:
                    return self._embedding_size
                return 1536
            case _:
                raise ValueError(
                    f"Unsupported model {self.model} for CohereEncoder. "
                    "Please specify the embedding size explicitly."
                )
