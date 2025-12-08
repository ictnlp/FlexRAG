from typing import Optional

import numpy as np

from flexrag.common import LOGGER_MANAGER, TIME_METER, configure

from .api_based_encoder import APIBasedEncoder, APIBasedEncoderConfig
from .encoder_base import ENCODERS

logger = LOGGER_MANAGER.get_logger("flexrag.models.ollama")


@configure
class OllamaEncoderConfig(APIBasedEncoderConfig):
    """Configuration for the OllamaEncoder.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server.
        Default is 'http://localhost:11434/'.
    :type base_url: str
    :param prompt: The prompt to use. Default is None.
    :type prompt: Optional[str]
    :param embedding_size: The size of the embeddings. Default is None.
    :type embedding_size: Optional[int]
    """

    model_name: Optional[str] = None
    base_url: str = "http://localhost:11434/"
    prompt: Optional[str] = None
    embedding_size: Optional[int] = None


@ENCODERS("ollama", config_class=OllamaEncoderConfig)
class OllamaEncoder(APIBasedEncoder):
    async def _create_client(self, config: OllamaEncoderConfig):
        from ollama import AsyncClient

        self._model_name = config.model_name
        self._prompt = config.prompt
        self._embedding_size = config.embedding_size
        return AsyncClient(host=config.base_url)

    @TIME_METER("encoder.ollama_encode")
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        if self._prompt:
            texts = [f"{self._prompt} {text}" for text in texts]
        r = await client.embed(model=self._model_name, input=texts)
        embeddings = np.array(r.embeddings)
        if self.embedding_size is None:
            return embeddings
        return embeddings[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int | None:
        return self._embedding_size
