from dataclasses import field
from typing import Any, Optional

import litellm
import numpy as np

from flexrag.common import ContentPart, configure, trace

from ..generators.litellm_generator import _image_part
from .encoder_base import ENCODERS, RemoteEncoderBase

litellm.suppress_debug_info = True


@configure
class LiteLLMEncoderConfig:
    """Configuration for LiteLLMEncoder.

    :param provider: LiteLLM provider prefix, e.g. ``openai`` or ``cohere``.
    :type provider: Optional[str]
    :param model_name: Provider-specific embedding model identifier without the provider prefix.
    :type model_name: Optional[str]
    :param api_key: API key passed to LiteLLM as ``api_key``. Defaults to None.
    :type api_key: Optional[str]
    :param base_url: Base URL passed to LiteLLM as ``api_base``. Defaults to None.
    :type base_url: Optional[str]
    :param api_version: Provider API version passed through to LiteLLM. Defaults to None.
    :type api_version: Optional[str]
    :param timeout: Request timeout in seconds. Defaults to None.
    :type timeout: Optional[float]
    :param proxy: Upstream proxy setting forwarded to LiteLLM. Defaults to None.
    :type proxy: Optional[str]
    :param embedding_size: Requested embedding dimension. Defaults to None.
    :type embedding_size: Optional[int]
    :param input_type: Provider-specific embedding input type. Defaults to None.
    :type input_type: Optional[str]
    :param extra_kwargs: Additional provider-specific LiteLLM embedding kwargs.
        Explicit top-level config fields take precedence over conflicting keys here.
    :type extra_kwargs: dict[str, Any]

    Example:

        >>> config = LiteLLMEncoderConfig(
        ...     provider="openai",
        ...     model_name="text-embedding-3-small",
        ...     embedding_size=1536,
        ...     input_type="search_document",
        ... )
        >>> encoder = LiteLLMEncoder(config)
    """

    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[float] = None
    proxy: Optional[str] = None
    embedding_size: Optional[int] = None
    input_type: Optional[str] = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@ENCODERS("litellm", config_class=LiteLLMEncoderConfig)
class LiteLLMEncoder(RemoteEncoderBase):
    def __init__(self, config: LiteLLMEncoderConfig):
        self._config = config
        self._embedding_size = config.embedding_size
        self._input_type = config.input_type
        self._client = self._build_client(config)
        return

    def _build_client(self, config: LiteLLMEncoderConfig):
        provider = (config.provider or "").strip()
        model_name = (config.model_name or "").strip()
        assert provider, "`provider` must be provided for LiteLLM models."
        assert model_name, "`model_name` must be provided for LiteLLM models."

        request_kwargs = dict(config.extra_kwargs)
        explicit_kwargs = {
            "api_key": config.api_key,
            "api_base": config.base_url,
            "api_version": config.api_version,
            "timeout": config.timeout,
            "proxy": config.proxy,
        }
        request_kwargs.update(
            {key: value for key, value in explicit_kwargs.items() if value is not None}
        )
        return {
            "model": f"{provider}/{model_name}",
            "request_kwargs": request_kwargs,
        }

    @trace("encoder.litellm_encode")
    async def _async_encode_batch(self, inputs: list[ContentPart]) -> np.ndarray:
        text_indices: list[int] = []
        text_inputs: list[str] = []
        image_indices: list[int] = []
        image_inputs: list[list[dict[str, Any]]] = []

        for idx, part in enumerate(inputs):
            part_type = part.get("type")
            if part_type == "text":
                text_indices.append(idx)
                text_inputs.append(part.get("text", ""))
                continue
            if part_type == "image":
                image_indices.append(idx)
                image_inputs.append([_image_part(part)])
                continue
            raise ValueError(
                "LiteLLMEncoder only supports text and image content blocks, "
                f"but got '{part_type}'."
            )

        results: list[np.ndarray | None] = [None] * len(inputs)
        if text_inputs:
            text_embeddings = await self._embed_inputs(text_inputs)
            for idx, embedding in zip(text_indices, text_embeddings, strict=True):
                results[idx] = embedding
        if image_inputs:
            image_embeddings = await self._embed_inputs(image_inputs)
            for idx, embedding in zip(image_indices, image_embeddings, strict=True):
                results[idx] = embedding

        ready_results = [result for result in results if result is not None]
        if len(ready_results) != len(results):
            raise RuntimeError("Some LiteLLMEncoder inputs did not produce embeddings.")
        return np.stack(ready_results, axis=0)

    async def _embed_inputs(self, inputs: list[Any]) -> np.ndarray:
        request_kwargs = dict(self._client["request_kwargs"])
        request_kwargs["model"] = self._client["model"]
        request_kwargs["input"] = inputs
        request_kwargs["input_type"] = self._input_type
        request_kwargs["dimensions"] = self._embedding_size
        response = await litellm.aembedding(**request_kwargs)
        data = response["data"] if isinstance(response, dict) else response.data
        embeddings = [
            item["embedding"] if isinstance(item, dict) else item.embedding
            for item in data
        ]
        array = np.array(embeddings)
        if self._embedding_size is None or array.size == 0:
            return array
        if array.shape[1] < self._embedding_size:
            raise ValueError(
                f"Embedding dimension {array.shape[1]} is smaller than requested {self._embedding_size}."
            )
        return array[:, : self._embedding_size]

    @property
    def embedding_size(self) -> int | None:
        return self._embedding_size
