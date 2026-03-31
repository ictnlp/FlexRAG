from dataclasses import field
from typing import Any, Optional

import litellm
import numpy as np

from flexrag.common import configure, trace

from .encoder_base import ENCODERS
from .remote_encoder_base import RemoteEncoderBase, RemoteEncoderBaseConfig

litellm.suppress_debug_info = True


@configure
class LiteLLMEncoderConfig(RemoteEncoderBaseConfig):
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
    async def _create_client(self, config: LiteLLMEncoderConfig):
        self._embedding_size = config.embedding_size
        self._input_type = config.input_type
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
    async def _async_encode_impl(self, client, texts: list[str]) -> np.ndarray:
        request_kwargs = dict(client["request_kwargs"])
        request_kwargs["model"] = client["model"]
        request_kwargs["input"] = texts
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
