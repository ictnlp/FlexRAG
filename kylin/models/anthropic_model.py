import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from kylin.prompt import ChatPrompt
from kylin.utils import TimeMeter

from .model_base import (
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
    Generators,
)

logger = logging.getLogger("AnthropicModel")


@dataclass
class AnthropicGeneratorConfig(GeneratorBaseConfig):
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "EMPTY")
    verbose: bool = False
    proxy: Optional[str] = None


@Generators("anthropic", config_class=AnthropicGeneratorConfig)
class AnthropicGenerator(GeneratorBase):
    def __init__(self, cfg: AnthropicGeneratorConfig) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            proxies=cfg.proxy,
        )
        self.model_name = cfg.model_name
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        return

    @TimeMeter("anthropic_generate")
    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses = []
        gen_cfg = self._get_options(generation_config, is_chat=True)
        for prompt in prompts:
            prompt = prompt.to_list()
            # as anthropic does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(generation_config.sample_num):
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=prompt,
                    **gen_cfg,
                )
                responses[-1].append(response.content[0].text)
        return responses

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

    @TimeMeter("anthropic_generate")
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses = []
        gen_cfg = self._get_options(generation_config)
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
