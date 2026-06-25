import asyncio
import os
from typing import Annotated, Optional

import numpy as np
from numpy import ndarray

from flexrag.utils import TIME_METER, Choices, configure

from .model_base import ENCODERS, EncoderBase, EncoderBaseConfig

# The embedding dimension produced by the Marengo family of models.
MARENGO_EMBEDDING_SIZE = 512


@configure
class TwelveLabsEncoderConfig(EncoderBaseConfig):
    """Configuration for TwelveLabsEncoder.

    TwelveLabs `Marengo <https://docs.twelvelabs.io/>`_ is a multimodal embedding
    model that maps text, image, audio, and video into a single 512-dimensional
    space. This encoder exposes the *text* embedding endpoint so that Marengo can
    be used as a FlexRAG query/passage encoder that is directly comparable with
    video embeddings produced by the same model.

    :param model: The Marengo model to use. Default is "marengo3.0".
    :type model: str
    :param api_key: The API key for the TwelveLabs API.
        If not provided, it will use the environment variable `TWELVELABS_API_KEY`.
        Defaults to None.
    :type api_key: Optional[str]
    """

    model: Annotated[str, Choices("marengo3.0")] = "marengo3.0"
    api_key: Optional[str] = None


@ENCODERS("twelvelabs", config_class=TwelveLabsEncoderConfig)
class TwelveLabsEncoder(EncoderBase):
    """Encode texts into the TwelveLabs Marengo 512-dimensional multimodal space."""

    def __init__(self, cfg: TwelveLabsEncoderConfig):
        super().__init__(cfg)
        try:
            from twelvelabs import AsyncTwelveLabs, TwelveLabs
        except ImportError:
            raise ImportError(
                "TwelveLabs is not installed. Please install it via `pip install twelvelabs`."
            )

        api_key = cfg.api_key or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for TwelveLabs is not provided. "
                "Please set it in the configuration or as an environment variable 'TWELVELABS_API_KEY'."
            )
        self.client = TwelveLabs(api_key=api_key)
        self.async_client = AsyncTwelveLabs(api_key=api_key)
        self.model = cfg.model
        return

    def _embed_one(self, text: str) -> list[float]:
        r = self.client.embed.create(model_name=self.model, text=text)
        return r.text_embedding.segments[0].float_

    @TIME_METER("twelvelabs_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        # The Marengo embed endpoint accepts a single text per request.
        embeddings = [self._embed_one(text) for text in texts]
        return np.array(embeddings)

    @TIME_METER("twelvelabs_encode")
    async def async_encode(self, texts: list[str]) -> ndarray:
        async def embed_one(text: str) -> list[float]:
            r = await self.async_client.embed.create(model_name=self.model, text=text)
            return r.text_embedding.segments[0].float_

        embeddings = await asyncio.gather(*[embed_one(text) for text in texts])
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        return MARENGO_EMBEDDING_SIZE
