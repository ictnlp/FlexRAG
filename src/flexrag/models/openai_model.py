import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from omegaconf import MISSING

from flexrag.prompt import ChatPrompt
from flexrag.utils import TIME_METER, LOGGER_MANAGER

from .model_base import (
    EncoderBase,
    ENCODERS,
    GenerationConfig,
    GeneratorBase,
    GENERATORS,
)

logger = LOGGER_MANAGER.get_logger("flexrag.models.openai")


@dataclass
class OpenAIConfig:
    """The Base Configuration for OpenAI Client.

    :param is_azure: Whether the model is hosted on Azure. Default is False.
    :type is_azure: bool
    :param model_name: The name of the model to use.
    :type model_name: str
    :param base_url: The base URL of the OpenAI API. Default is None.
    :type base_url: Optional[str]
    :param api_key: The API key for OpenAI. Default is os.environ.get("OPENAI_API_KEY", "EMPTY").
    :type api_key: str
    :param api_version: The API version to use. Default is "2024-07-01-preview".
    :type api_version: str
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param proxy: The proxy to use for the HTTP client. Default is None.
    :type proxy: Optional[str]
    """

    is_azure: bool = False
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_version: str = "2024-07-01-preview"
    verbose: bool = False
    proxy: Optional[str] = None


@dataclass
class OpenAIGeneratorConfig(OpenAIConfig):
    """Configuration for OpenAI Generator.

    :param allow_parallel: Whether to allow parallel generation. Default is True.
    :type allow_parallel: bool
    """

    allow_parallel: bool = True


@dataclass
class OpenAIEncoderConfig(OpenAIConfig):
    """Configuration for OpenAI Encoder.

    :param is_azure: Whether the model is hosted on Azure. Default is False.
    :type is_azure: bool
    """

    dimension: Optional[int] = None


@GENERATORS("openai", config_class=OpenAIGeneratorConfig)
class OpenAIGenerator(GeneratorBase):
    def __init__(self, cfg: OpenAIGeneratorConfig) -> None:
        from openai import OpenAI, AzureOpenAI

        # prepare proxy
        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None

        # prepare client
        if cfg.is_azure:
            self.client = AzureOpenAI(
                api_key=cfg.api_key,
                api_version=cfg.api_version,
                azure_endpoint=cfg.base_url,
                http_client=httpx_client,
            )
        else:
            self.client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                http_client=httpx_client,
            )

        # set logger
        self.allow_parallel = cfg.allow_parallel
        self.model_name = cfg.model_name
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)

        # check client
        self._check()
        return

    @TIME_METER("openai_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        gen_cfg = self._get_options(generation_config)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prompt: [
                        r.message.content
                        for r in self.client.chat.completions.create(
                            model=self.model_name,
                            messages=prompt.to_list(),
                            **gen_cfg,
                        ).choices
                    ],
                    prompts,
                )
                responses = list(responses)
        else:
            responses = []
            for prompt in prompts:
                prompt = prompt.to_list()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt,
                    **gen_cfg,
                )
                responses.append([i.message.content for i in response.choices])
        return responses

    @TIME_METER("openai_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        tasks = []
        gen_cfg = self._get_options(generation_config)
        for prompt in prompts:
            prompt = prompt.to_list()
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        self.client.chat.completions.create,
                        model=self.model_name,
                        messages=prompt,
                        **gen_cfg,
                    )
                )
            )
        responses = [[i.message.content for i in (await r).choices] for r in tasks]
        return responses

    @TIME_METER("openai_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        gen_cfg = self._get_options(generation_config)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prefix: [
                        r.text
                        for r in self.client.completions.create(
                            model=self.model_name,
                            prompt=prefix,
                            **gen_cfg,
                        ).choices
                    ],
                    prefixes,
                )
                responses = list(responses)
        else:
            responses = []
            for prefix in prefixes:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prefix,
                    **gen_cfg,
                )
                responses.append([i.text for i in response.choices])
        return responses

    @TIME_METER("openai_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        tasks = []
        gen_cfg = self._get_options(generation_config)
        for prefix in prefixes:
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        self.client.completions.create,
                        model=self.model_name,
                        prompt=prefix,
                        **gen_cfg,
                    )
                )
            )
        responses = [[i.text for i in (await r).choices] for r in tasks]
        return responses

    def _get_options(self, generation_config: GenerationConfig) -> dict:
        if "llama-3" in self.model_name.lower():
            extra_body = {"stop_token_ids": [128009]}  # hotfix for llama-3
        else:
            extra_body = None
        return {
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "max_tokens": generation_config.max_new_tokens,
            "top_p": generation_config.top_p,
            "n": generation_config.sample_num,
            "extra_body": extra_body,
            "stop": list(generation_config.stop_str),
        }

    def _check(self):
        model_lists = [i.id for i in self.client.models.list().data]
        assert self.model_name in model_lists, f"Model {self.model_name} not found"


@ENCODERS("openai", config_class=OpenAIEncoderConfig)
class OpenAIEncoder(EncoderBase):
    def __init__(self, cfg: OpenAIEncoderConfig) -> None:
        from openai import OpenAI, AzureOpenAI

        # prepare proxy
        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None

        # prepare client
        if cfg.is_azure:
            self.client = AzureOpenAI(
                api_key=cfg.api_key,
                api_version=cfg.api_version,
                azure_endpoint=cfg.base_url,
                http_client=httpx_client,
            )
        else:
            self.client = OpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                http_client=httpx_client,
            )

        # set logger
        self.model_name = cfg.model_name
        self.dimension = cfg.dimension
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)

        # check client
        self._check()
        return

    @TIME_METER("openai_encode")
    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.dimension is None:
            r = self.client.embeddings.create(model=self.model_name, input=texts)
        else:
            r = self.client.embeddings.create(
                model=self.model_name, input=texts, dimensions=self.dimension
            )
        embeddings = [i.embedding for i in r.data]
        return np.array(embeddings)

    @TIME_METER("openai_encode")
    async def async_encode(self, texts: list[str]) -> np.ndarray:
        r = await asyncio.to_thread(
            self.client.embeddings.create,
            model=self.model_name,
            input=texts,
            dimensions=self.dimension,
        )
        embeddings = [i.embedding for i in r.data]
        return np.array(embeddings)

    @property
    def embedding_size(self):
        if self.dimension is None:
            return len(
                self.client.embeddings.create(model=self.model_name, input="test")
                .data[0]
                .embedding
            )
        return self.dimension

    def _check(self):
        model_lists = [i.id for i in self.client.models.list().data]
        assert self.model_name in model_lists, f"Model {self.model_name} not found"
