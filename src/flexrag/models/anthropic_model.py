import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from flexrag.common import LOGGER_MANAGER, TIME_METER, configure
from flexrag.common.prompt import ChatPrompt

from .model_base import GENERATORS, GenerationConfig, GeneratorBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.anthropic")


@configure
class AnthropicGeneratorConfig:
    """Configuration for AnthropicGenerator.

    :param model_name: The name of the model. Required.
    :type model_name: str
    :param base_url: The base url of the API. Defaults to None.
    :type base_url: Optional[str]
    :param api_key: The API key. Defaults to os.environ.get("ANTHROPIC_API_KEY", "EMPTY").
    :type api_key: str
    :param verbose: Whether to output verbose logs. Defaults to False.
    :type verbose: bool
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    :param max_concurrency: The maximum number of concurrent generation requests. Defaults to 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "EMPTY")
    verbose: bool = False
    proxy: Optional[str] = None
    max_concurrency: int = 1


@GENERATORS("anthropic", config_class=AnthropicGeneratorConfig)
class AnthropicGenerator(GeneratorBase):
    def __init__(self, cfg: AnthropicGeneratorConfig) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        # initialize the client
        if cfg.proxy is not None:
            client = httpx.Client(proxies=cfg.proxy)
        else:
            client = None
        self.client = Anthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            http_client=client,
        )
        self.async_client = AsyncAnthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            http_client=client,
        )

        # set arguments
        assert cfg.model_name is not None, "model_name must be provided"
        self.model_name = cfg.model_name
        self.max_concurrency = cfg.max_concurrency

        # set logger
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        return

    @TIME_METER("generator.anthropic_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as anthropic does not support sample_num, we sample multiple times
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        gen_cfg = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        def _create(prompt: ChatPrompt) -> str:
            r = self.client.messages.create(
                model=self.model_name,
                messages=prompt.to_list(),
                **gen_cfg,
            )
            return r.content[0].text

        tasks = prompts * sample_num
        if self.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                responses = list(pool.map(_create, tasks))
        else:
            responses = []
            for task in tasks:
                responses.append(_create(task))
        responses: list[list[str]] = [
            responses[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prompts))
        ]
        return responses

    @TIME_METER("generator.anthropic_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        gen_cfg = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(prompt: ChatPrompt) -> str:
            async with semaphore:
                r = await self.async_client.messages.create(
                    model=self.model_name,
                    messages=prompt.to_list(),
                    **gen_cfg,
                )
                return r.content[0].text

        tasks = prompts * sample_num
        tasks = [asyncio.create_task(_create(prompt)) for prompt in tasks]
        responses = await asyncio.gather(*tasks)
        responses: list[list[str]] = [
            responses[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prompts))
        ]
        return responses

    @TIME_METER("generator.anthropic_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        raise NotImplementedError("The Anthropic text completion API is deprecated.")

    @TIME_METER("generator.anthropic_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        raise NotImplementedError("The Anthropic text completion API is deprecated.")

    def _get_options(self, generation_config: GenerationConfig) -> dict:
        return {
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "max_tokens": generation_config.max_new_tokens,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
            "stop_sequences": generation_config.stop_str,
        }
