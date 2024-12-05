import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

from concurrent.futures import ThreadPoolExecutor
from omegaconf import MISSING

from librarian.prompt import ChatPrompt
from librarian.utils import TIME_METER, LOGGER_MANAGER

from .model_base import (
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
    GENERATORS,
)

logger = LOGGER_MANAGER.get_logger("librarian.models.anthropic")


@dataclass
class AnthropicGeneratorConfig(GeneratorBaseConfig):
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "EMPTY")
    verbose: bool = False
    proxy: Optional[str] = None
    allow_parallel: bool = True


@GENERATORS("anthropic", config_class=AnthropicGeneratorConfig)
class AnthropicGenerator(GeneratorBase):
    def __init__(self, cfg: AnthropicGeneratorConfig) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            proxies=cfg.proxy,
        )
        self.model_name = cfg.model_name
        self.allow_parallel = cfg.allow_parallel
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        return

    @TIME_METER("anthropic_generate")
    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as anthropic does not support sample_num, we sample multiple times
        gen_cfg = self._get_options(generation_config, is_chat=True)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prompt: [
                        self.client.messages.create(
                            model=self.model_name,
                            messages=prompt.to_list(),
                            **gen_cfg,
                        )
                        .content[0]
                        .text
                        for _ in range(generation_config.sample_num)
                    ],
                    prompts,
                )
                responses = list(responses)
        else:
            responses: list[list[str]] = []
            for prompt in prompts:
                prompt = prompt.to_list()
                responses.append([])
                for _ in range(generation_config.sample_num):
                    response = self.client.messages.create(
                        model=self.model_name,
                        messages=prompt,
                        **gen_cfg,
                    )
                    responses[-1].append(response.content[0].text)
        return responses

    @TIME_METER("anthropic_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        gen_cfg = self._get_options(generation_config, is_chat=True)
        tasks = []
        for prompt in prompts:
            prompt = prompt.to_list()
            # as anthropic does not support sample_num, we sample multiple times
            tasks.append([])
            for _ in range(generation_config.sample_num):
                tasks[-1].append(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self.client.messages.create,
                            model=self.model_name,
                            messages=prompt,
                            **gen_cfg,
                        )
                    )
                )
        responses = [
            [(await task).content[0].text for task in task_list] for task_list in tasks
        ]
        return responses

    @TIME_METER("anthropic_generate")
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        gen_cfg = self._get_options(generation_config)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prefix: [
                        self.client.completions.create(
                            model=self.model_name,
                            prompt=prefix,
                            **gen_cfg,
                        ).completion
                        for _ in range(generation_config.sample_num)
                    ],
                    prefixes,
                )
                responses = list(responses)
        else:
            responses: list[list[str]] = []
            for prefix in prefixes:
                # as anthropic does not support sample_num, we sample multiple times
                responses.append([])
                for _ in range(generation_config.sample_num):
                    response = self.client.completions.create(
                        model=self.model_name,
                        prompt=prefix,
                        **gen_cfg,
                    )
                    responses[-1].append(response.completion)
        return responses

    @TIME_METER("anthropic_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        tasks = []
        gen_cfg = self._get_options(generation_config)
        for prefix in prefixes:
            # as anthropic does not support sample_num, we sample multiple times
            tasks.append([])
            for _ in range(generation_config.sample_num):
                tasks[-1].append(
                    asyncio.create_task(
                        await asyncio.to_thread(
                            self.client.completions.create,
                            model=self.model_name,
                            prompt=prefix,
                            **gen_cfg,
                        )
                    )
                )
        responses = [
            [(await task).completion for task in task_list] for task_list in tasks
        ]
        return responses

    def _get_options(
        self, generation_config: GenerationConfig, is_chat: bool = False
    ) -> dict:
        if is_chat:
            return {
                "temperature": (
                    generation_config.temperature
                    if generation_config.do_sample
                    else 0.0
                ),
                "max_tokens": generation_config.max_new_tokens,
                "top_p": generation_config.top_p,
                "top_k": generation_config.top_k,
            }
        return {
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "max_tokens_to_sample": generation_config.max_new_tokens,
            "top_p": generation_config.top_p,
            "top_k": generation_config.top_k,
        }
