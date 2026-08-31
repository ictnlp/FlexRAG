from dataclasses import field
from typing import Any, Optional

import litellm
import numpy as np

from flexrag.common import configure, trace
from flexrag.models.litellm_client import LiteLLMRerankClient

from .ranker_base import RANKERS
from .remote_ranker_base import RemoteRankerBase, RemoteRankerBaseConfig

litellm.suppress_debug_info = True


@configure
class LiteLLMRankerConfig(RemoteRankerBaseConfig):
    """Configuration for LiteLLMRanker.

    :param provider: LiteLLM provider prefix, e.g. ``cohere`` or ``voyage``.
    :type provider: Optional[str]
    :param model_name: Provider-specific rerank model identifier without the provider prefix.
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
    :param extra_kwargs: Additional provider-specific LiteLLM rerank kwargs.
        Explicit top-level config fields take precedence over conflicting keys here.
    :type extra_kwargs: dict[str, Any]
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
    async def _create_client(self, config: LiteLLMRankerConfig) -> LiteLLMRerankClient:
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
        return await LiteLLMRerankClient.create(
            provider=provider,
            model_name=model_name,
            request_kwargs=request_kwargs,
            timeout=config.timeout,
        )

    async def _close_client(self, client: LiteLLMRerankClient) -> None:
        await client.close()
        return

    @trace("ranker.litellm_rerank")
    async def _async_rank_impl(
        self, client, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        response = await litellm.arerank(
            model=client.model,
            query=query,
            documents=candidates,
            top_n=len(candidates),
            return_documents=False,
            **client.request_kwargs,
        )
        scores = np.zeros(len(candidates))
        for result in response.results:
            if isinstance(result, dict):
                index = result["index"]
                relevance_score = result["relevance_score"]
            else:
                index = result.index
                relevance_score = result.relevance_score
            scores[index] = relevance_score
        return None, scores
