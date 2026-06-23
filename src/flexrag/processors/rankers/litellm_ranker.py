from dataclasses import field
from typing import Any, Optional

import litellm
import numpy as np

from flexrag.common import configure, trace

from .ranker_base import RANKERS, RankerBaseConfig, RemoteRankerBase

litellm.suppress_debug_info = True


@configure
class LiteLLMRankerConfig(RankerBaseConfig):
    """Configuration for LiteLLMRanker.

    :param provider: LiteLLM provider prefix, e.g. ``cohere`` or ``voyage``.
    :param model_name: Provider-specific rerank model identifier without the
        provider prefix.
    :param api_key: API key passed to LiteLLM as ``api_key``. Defaults to None.
    :param base_url: Base URL passed to LiteLLM as ``api_base``. Defaults to None.
    :param api_version: Provider API version passed through to LiteLLM. Defaults
        to None.
    :param timeout: Request timeout in seconds. Defaults to None.
    :param proxy: Upstream proxy setting forwarded to LiteLLM. Defaults to None.
    :param extra_kwargs: Additional provider-specific LiteLLM rerank kwargs.
        Explicit top-level config fields take precedence over conflicting keys here.
    """

    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[float] = None
    proxy: Optional[str] = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@RANKERS("litellm", config_class=LiteLLMRankerConfig)
class LiteLLMRanker(RemoteRankerBase):
    """Raw LiteLLM rerank implementation.

    The class owns provider request construction and the LiteLLM rerank call.
    It exposes the direct-use ranker interface inherited from
    ``RemoteRankerBase``.
    """

    def __init__(self, config: LiteLLMRankerConfig) -> None:
        super().__init__(config)
        provider = (config.provider or "").strip()
        model_name = (config.model_name or "").strip()
        assert provider, "`provider` must be provided for LiteLLM rankers."
        assert model_name, "`model_name` must be provided for LiteLLM rankers."

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
        self._model = f"{provider}/{model_name}"
        self._request_kwargs = request_kwargs
        return

    @trace("ranker.litellm_rerank")
    async def _async_rank_batch(
        self,
        query: str,
        candidates: list[str],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        response = await litellm.arerank(
            model=self._model,
            query=query,
            documents=candidates,
            top_n=len(candidates),
            return_documents=False,
            **self._request_kwargs,
        )
        scores = np.zeros(len(candidates))
        for result in response.results:
            scores[result.index] = result.relevance_score
        return None, scores
