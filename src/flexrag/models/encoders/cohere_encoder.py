import os
from typing import Annotated, Optional

import httpx
import numpy as np

from flexrag.common import TIME_METER, Choices, configure

from .api_based_encoder import APIBasedEncoder, APIBasedEncoderConfig
from .encoder_base import ENCODERS


@configure
class CohereEncoderConfig(APIBasedEncoderConfig):
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
    :param api_key: The API key for the Cohere API.
        If not provided, it will use the environment variable `COHERE_API_KEY`.
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
    api_key: str = os.environ.get("COHERE_API_KEY")
    base_url: Optional[str] = None
    proxy: Optional[str] = None


@ENCODERS("cohere", config_class=CohereEncoderConfig)
class CohereEncoder(APIBasedEncoder):
    async def _create_client(self, config: CohereEncoderConfig):
        from cohere import AsyncClientV2

        self._model_name = config.model
        self._dimension = int(config.embedding_size)
        self._input_type = config.input_type

        if config.proxy is not None:
            httpx_client = httpx.Client(proxies=config.proxy)
        else:
            httpx_client = None
        return AsyncClientV2(
            api_key=config.api_key,
            base_url=config.base_url,
            httpx_client=httpx_client,
        )

    @TIME_METER("encoder.cohere_encode")
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        embed_dim = self.embedding_size if self._model_name == "embed-v4.0" else None
        r = await client.embed(
            texts=texts,
            model=self._model_name,
            input_type=self._input_type,
            embedding_types=["float"],
            output_dimension=embed_dim,
        )
        embeddings = r.embeddings.float
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        match self._model_name:
            case "embed-multilingual-light-v3.0":
                return 384
            case "embed-multilingual-v3.0":
                return 1024
            case "embed-english-light-v3.0":
                return 384
            case "embed-english-v3.0":
                return 1024
            case "embed-v4.0":
                if self._dimension is not None:
                    return self._dimension
                return 1536
            case _:
                raise ValueError(
                    f"Unsupported model {self._model_name} for CohereEncoder. "
                    "Please specify the embedding size explicitly."
                )
