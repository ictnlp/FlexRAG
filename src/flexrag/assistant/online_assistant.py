import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import httpx
from omegaconf import MISSING

from flexrag.common_dataclass import RetrievedContext
from flexrag.models import GenerationConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import LOGGER_MANAGER, Choices

from .assistant import ASSISTANTS, AssistantBase

logger = LOGGER_MANAGER.get_logger("flexrag.assistant")


@dataclass
class JinaDeepSearchConfig:
    """The configuration for the Jina DeepSearch Assistant.

    :param prompt_path: The path to the prompt file. Defaults to None.
    :type prompt_path: str, optional
    :param use_history: Whether to save the chat history for multi-turn conversation. Defaults to False.
    :type use_history: bool, optional
    :param base_url: The base URL of the API. Defaults to "https://deepsearch.jina.ai/v1/chat/completions".
    :type base_url: str
    :param api_key: The API key. Defaults to os.getenv("JINA_API_KEY", MISSING).
    :type api_key: str
    :param model: The model to use. Defaults to "jina-deepsearch-v1".
    :type model: str
    :param reasoning_effort: The reasoning effort. Defaults to "medium".
        Available choices are "low", "medium", "high".
    :type reasoning_effort: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: str, optional
    :param timeout: The timeout for the API call. Defaults to 10.
        Note that the deepsearch API may take a long time to respond.
    :type timeout: int
    """

    prompt_path: Optional[str] = None
    use_history: bool = False
    base_url: str = "https://deepsearch.jina.ai/v1"
    api_key: str = os.getenv("JINA_API_KEY", MISSING)
    model: str = "jina-deepsearch-v1"
    reasoning_effort: Choices(["low", "medium", "high"]) = "medium"  # type: ignore
    proxy: Optional[str] = None
    timeout: int = 10


@ASSISTANTS("jina_deepsearch", config_class=JinaDeepSearchConfig)
class JinaDeepSearch(AssistantBase):
    """The Jina DeepSearch Assistant (https://jina.ai/deepsearch/)."""

    def __init__(self, cfg: JinaDeepSearchConfig):
        # prepare prompts
        if cfg.prompt_path is not None:
            self.prompt = ChatPrompt.from_json(cfg.prompt_path)
        else:
            self.prompt = ChatPrompt()
        if cfg.use_history:
            self.history_prompt = deepcopy(self.prompt)
        else:
            self.history_prompt = None

        # prepare client
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            proxy=cfg.proxy,
            follow_redirects=True,
            timeout=cfg.timeout,
        )

        # prepare data template
        self.data_template = {
            "model": cfg.model,
            "messages": [],
            "reasoning_effort": cfg.reasoning_effort,
            "stream": False,
        }
        return

    def answer(self, question: str) -> tuple[str, None, dict[str, ChatPrompt]]:
        # prepare prompt
        if self.history_prompt is not None:
            prompt = self.history_prompt
        else:
            prompt = deepcopy(self.prompt)

        prompt.update(ChatTurn(role="user", content=question))

        # prepare data
        data = deepcopy(self.data_template)
        data["messages"] = prompt.to_list()

        # generate response
        response = self.client.post("chat/completions", json=data)
        response.raise_for_status()
        response = response.json()["choices"][0]["message"]["content"]

        # update the prompt
        prompt.update(ChatTurn(role="assistant", content=response))
        return response, None, {"prompt": prompt}

    def clear_history(self) -> None:
        self.history_prompt = deepcopy(self.prompt)
        return


@dataclass
class PerplexityAssistantConfig(GenerationConfig):
    """The configuration for the PerplexityAI Assistant.

    :param prompt_path: The path to the prompt file. Defaults to None.
    :type prompt_path: str, optional
    :param use_history: Whether to save the chat history for multi-turn conversation. Defaults to False.
    :type use_history: bool, optional
    :param base_url: The base URL of the API. Defaults to "https://api.perplexity.ai/chat/completions".
    :type base_url: str
    :param api_key: The API key. Defaults to os.getenv("PERPLEXITY_API_KEY", MISSING).
    :type api_key: str
    :param model: The model to use. Defaults to "sonar".
    :type model: str
    :param search_domain_filter: Given a list of domains, limit the citations used by the online model to URLs from the specified domains.
        Defaults to []. Only available to users in Tier-3.
    :type search_domain_filter: list[str]
    :param search_recency_filter: Returns search results within the specified time interval.
        Defaults to None. Available choices are "month", "week", "day", "hour".
    :type search_recency_filter: str, optional
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: str, optional
    :param timeout: The timeout for the API call. Defaults to 10.
    :type timeout: int
    """

    prompt_path: Optional[str] = None
    use_history: bool = False
    base_url: str = "https://api.perplexity.ai"
    api_key: str = os.environ.get("PERPLEXITY_API_KEY", MISSING)
    model: str = "sonar"
    search_domain_filter: list[str] = field(default_factory=list)
    search_recency_filter: Optional[Choices(["month", "week", "day", "hour"])] = None  # type: ignore
    proxy: Optional[str] = None
    timeout: int = 10


@ASSISTANTS("perplexity", config_class=PerplexityAssistantConfig)
class PerplexityAssistant(AssistantBase):
    """The PerplexityAI Assistant (https://www.perplexity.ai)."""

    def __init__(self, cfg: PerplexityAssistantConfig):
        # load prompts
        if cfg.prompt_path is not None:
            self.prompt = ChatPrompt.from_json(cfg.prompt_path)
        else:
            self.prompt = ChatPrompt()
        if cfg.use_history:
            self.history_prompt = deepcopy(self.prompt)
        else:
            self.history_prompt = None

        # prepare client
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            proxy=cfg.proxy,
            follow_redirects=True,
            timeout=cfg.timeout,
        )

        # prepare message template
        self.data_template = {
            "model": cfg.model,
            "messages": [],
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
        }
        if cfg.search_domain_filter is not None:
            self.data_template["search_domain_filter"] = cfg.search_domain_filter
        if cfg.search_recency_filter is not None:
            self.data_template["search_recency_filter"] = cfg.search_recency_filter
        return

    def answer(
        self, question: str
    ) -> tuple[str, list[RetrievedContext], dict[str, ChatPrompt]]:
        # prepare prompt
        if self.history_prompt is not None:
            prompt = self.history_prompt
        else:
            prompt = deepcopy(self.prompt)

        prompt.update(ChatTurn(role="user", content=question))

        # prepare data
        data = deepcopy(self.data_template)
        data["messages"] = prompt.to_list()

        # generate response
        response = self.client.post("chat/completions", json=data)
        response.raise_for_status()
        r = response.json()["choices"][0]["message"]["content"]
        contexts = [
            RetrievedContext(source=i, retriever="perplexity", query=question)
            for i in response.json()["citations"]
        ]

        # update the prompt
        prompt.update(ChatTurn(role="assistant", content=r))
        return r, contexts, {"prompt": prompt}

    def clear_history(self) -> None:
        self.history_prompt = deepcopy(self.prompt)
        return
