from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import field
from typing import Annotated, Any, Optional

import httpx

from flexrag.common import Choices, configure
from flexrag.common.dataclasses import ChatMessages, ChatTurn, RetrievedContext
from flexrag.models import GenerationConfig

from .assistant_base import ASSISTANTS, AssistantBase, AssistantResult


@configure
class JinaDeepSearchConfig:
    """Configuration for the Jina DeepSearch Assistant.

    :param base_url: Jina DeepSearch API base URL.
    :param api_key: API key. Falls back to ``JINA_API_KEY``.
    :param model: Model name sent to the API.
    :param reasoning_effort: Provider reasoning-effort setting.
    :param proxy: Optional HTTP proxy.
    :param timeout: Request timeout in seconds.
    """

    base_url: str = "https://deepsearch.jina.ai/v1"
    api_key: Optional[str] = None
    model: str = "jina-deepsearch-v1"
    reasoning_effort: Annotated[str, Choices("low", "medium", "high")] = "medium"
    proxy: Optional[str] = None
    timeout: int = 10


@ASSISTANTS("jina_deepsearch", config_class=JinaDeepSearchConfig)
class JinaDeepSearch(AssistantBase):
    """Stateless QA assistant backed by Jina DeepSearch.

    See https://jina.ai/deepsearch/.
    """

    def __init__(self, config: JinaDeepSearchConfig) -> None:
        super().__init__()
        api_key = config.api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise ValueError(
                "Jina API key is required; set config.api_key or JINA_API_KEY"
            )
        self._client_kwargs: dict[str, Any] = {
            "base_url": config.base_url,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            "proxy": config.proxy,
            "follow_redirects": True,
            "timeout": config.timeout,
        }
        self._data_template = {
            "model": config.model,
            "messages": [],
            "reasoning_effort": config.reasoning_effort,
            "stream": False,
        }
        self._client: httpx.AsyncClient | None = None

    async def _start_episode(self) -> None:
        self._client = httpx.AsyncClient(**self._client_kwargs)

    async def _finish_episode(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _answer(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool,
    ) -> AssistantResult:
        self._validate_qa_request(retrieve)
        data = deepcopy(self._data_template)
        data["messages"] = self._provider_messages(messages)
        response = await self._require_client().post("chat/completions", json=data)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return AssistantResult(
            response=ChatTurn(role="assistant", content=content),
            metadata={"prompt": deepcopy(messages)},
        )

    @staticmethod
    def _validate_qa_request(retrieve: bool) -> None:
        if not retrieve:
            raise ValueError("JinaDeepSearch requires retrieve=True")

    @staticmethod
    def _provider_messages(messages: ChatMessages) -> list[dict[str, Any]]:
        return [
            {"role": turn.role, "content": deepcopy(turn.content)} for turn in messages
        ]

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("JinaDeepSearch requires an active episode")
        return self._client


@configure
class PerplexityAssistantConfig(GenerationConfig):
    """Configuration for the Perplexity QA Assistant.

    :param base_url: Perplexity API base URL.
    :param api_key: API key. Falls back to ``PERPLEXITY_API_KEY``.
    :param model: Model name sent to the API.
    :param search_domain_filter: Optional domains used to constrain search.
    :param search_recency_filter: Optional search-recency constraint.
    :param proxy: Optional HTTP proxy.
    :param timeout: Request timeout in seconds.
    """

    base_url: str = "https://api.perplexity.ai"
    api_key: Optional[str] = None
    model: str = "sonar"
    search_domain_filter: list[str] = field(default_factory=list)
    search_recency_filter: Annotated[
        str, Choices("month", "week", "day", "hour", "none")
    ] = "none"
    proxy: Optional[str] = None
    timeout: int = 10


@ASSISTANTS("perplexity", config_class=PerplexityAssistantConfig)
class PerplexityAssistant(AssistantBase):
    """Stateless QA assistant backed by Perplexity.

    See https://docs.perplexity.ai/.
    """

    def __init__(self, config: PerplexityAssistantConfig) -> None:
        super().__init__()
        api_key = config.api_key or os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError(
                "Perplexity API key is required; set config.api_key or "
                "PERPLEXITY_API_KEY"
            )
        self._client_kwargs: dict[str, Any] = {
            "base_url": config.base_url,
            "headers": {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            "proxy": config.proxy,
            "follow_redirects": True,
            "timeout": config.timeout,
        }
        self._data_template: dict[str, Any] = {
            "model": config.model,
            "messages": [],
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
        }
        if config.search_domain_filter:
            self._data_template["search_domain_filter"] = config.search_domain_filter
        if config.search_recency_filter != "none":
            self._data_template["search_recency_filter"] = config.search_recency_filter
        self._client: httpx.AsyncClient | None = None

    async def _start_episode(self) -> None:
        self._client = httpx.AsyncClient(**self._client_kwargs)

    async def _finish_episode(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _answer(
        self,
        messages: ChatMessages,
        *,
        retrieve: bool,
    ) -> AssistantResult:
        self._validate_qa_request(retrieve)
        data = deepcopy(self._data_template)
        data["messages"] = JinaDeepSearch._provider_messages(messages)
        response = await self._require_client().post("chat/completions", json=data)
        response.raise_for_status()
        payload = response.json()
        query = messages[-1].text_content
        contexts = [
            RetrievedContext(
                context_id=citation,
                data={"text": citation},
                source=citation,
                retriever="perplexity",
                query=query,
            )
            for citation in payload.get("citations", [])
        ]
        return AssistantResult(
            response=ChatTurn(
                role="assistant",
                content=payload["choices"][0]["message"]["content"],
            ),
            contexts=contexts,
            metadata={"prompt": deepcopy(messages)},
        )

    @staticmethod
    def _validate_qa_request(retrieve: bool) -> None:
        if not retrieve:
            raise ValueError("PerplexityAssistant requires retrieve=True")

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("PerplexityAssistant requires an active episode")
        return self._client
