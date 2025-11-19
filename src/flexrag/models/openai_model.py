import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
import numpy as np
from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, Omit, OpenAI

from flexrag.common import LOGGER_MANAGER, TIME_METER, configure
from flexrag.common.prompt import ChatPrompt

from .model_base import (
    ENCODERS,
    GENERATORS,
    EncoderBase,
    EncoderBaseConfig,
    GenerationConfig,
    GeneratorBase,
)

logger = LOGGER_MANAGER.get_logger("flexrag.models.openai")


@configure
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
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_version: str = "2024-07-01-preview"
    verbose: bool = False
    proxy: Optional[str] = None


@configure
class OpenAIGeneratorConfig(OpenAIConfig):
    """Configuration for OpenAI Generator.

    :param max_concurrency: The maximum number of concurrent generation requests. Default is 1.
    :type max_concurrency: int
    """

    max_concurrency: int = 1


@GENERATORS("openai", config_class=OpenAIGeneratorConfig)
class OpenAIGenerator(GeneratorBase):
    def __init__(self, cfg: OpenAIGeneratorConfig) -> None:
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
            self.async_client = AsyncAzureOpenAI(
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
            self.async_client = AsyncOpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                http_client=httpx_client,
            )

        # set logger
        assert cfg.model_name is not None, "`model_name` must be provided"
        self.model_name = cfg.model_name
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)

        # set semaphore
        self.max_concurrency = cfg.max_concurrency

        # check client
        self._check()
        return

    @TIME_METER("generator.openai_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        gen_cfg = self._get_options(generation_config)

        def _create(prompt: ChatPrompt) -> list[str]:
            prompt_messages = prompt.to_list()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt_messages,
                **gen_cfg,
            )
            return [choice.message.content for choice in response.choices]

        if self.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                responses = pool.map(_create, prompts)
                responses = list(responses)
        else:
            responses = []
            for prompt in prompts:
                responses.append(_create(prompt))
        return responses

    @TIME_METER("generator.openai_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        gen_cfg = self._get_options(generation_config)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(prompt: ChatPrompt) -> list[str]:
            prompt_messages = prompt.to_list()
            async with semaphore:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=prompt_messages,
                    **gen_cfg,
                )
            return [choice.message.content for choice in response.choices]

        tasks = [asyncio.create_task(_create(prompt)) for prompt in prompts]
        return await asyncio.gather(*tasks)

    @TIME_METER("generator.openai_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [prefixes] if not isinstance(prefixes, list) else prefixes
        gen_cfg = self._get_options(generation_config)

        def _create(prefix: str) -> list[str]:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prefix,
                **gen_cfg,
            )
            return [i.text for i in response.choices]

        if self.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                responses = pool.map(_create, prefixes)
                responses = list(responses)
        else:
            responses = []
            for prefix in prefixes:
                responses.append(_create(prefix))
        return responses

    @TIME_METER("generator.openai_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [prefixes] if not isinstance(prefixes, list) else prefixes
        gen_cfg = self._get_options(generation_config)

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(prefix: str) -> list[str]:
            async with semaphore:
                response = await self.async_client.completions.create(
                    model=self.model_name,
                    prompt=prefix,
                    **gen_cfg,
                )
            return [i.text for i in response.choices]

        tasks = [asyncio.create_task(_create(prefix)) for prefix in prefixes]
        return await asyncio.gather(*tasks)

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


@configure
class OpenAIEncoderConfig(OpenAIConfig, EncoderBaseConfig):
    """Configuration for OpenAI Encoder.

    :param embedding_size: The size of the embedding vector.
        If None, it will be determined from the model.
        Default is None.
    :type embedding_size: Optional[int]
    """

    embedding_size: Optional[int] = None


@ENCODERS("openai", config_class=OpenAIEncoderConfig)
class OpenAIEncoder(EncoderBase):
    def __init__(self, cfg: OpenAIEncoderConfig) -> None:
        super().__init__(cfg)
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
            self.async_client = AsyncAzureOpenAI(
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
            self.async_client = AsyncOpenAI(
                api_key=cfg.api_key,
                base_url=cfg.base_url,
                http_client=httpx_client,
            )

        # set logger
        assert cfg.model_name is not None, "`model_name` must be provided"
        self.model_name = cfg.model_name
        self.dimension = cfg.embedding_size
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)

        # check client
        self._check()
        return

    @TIME_METER("encoder.openai_encode")
    def _encode(self, texts: list[str]) -> np.ndarray:
        texts = [texts] if not isinstance(texts, list) else texts
        dimension = self.dimension if self.dimension else Omit()
        r = self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=dimension,
            encoding_format="float",
        )
        embeddings = [i.embedding for i in r.data]
        return np.array(embeddings)

    @TIME_METER("encoder.openai_encode")
    async def async_encode(self, texts: list[str]) -> np.ndarray:
        texts = [texts] if not isinstance(texts, list) else texts
        dimension = self.dimension if self.dimension else Omit()
        r = await self.async_client.embeddings.create(
            model=self.model_name,
            input=texts,
            dimensions=dimension,
            encoding_format="float",
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
