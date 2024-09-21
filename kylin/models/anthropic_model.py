import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
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


@Generators("anthropic", config_class=AnthropicGeneratorConfig)
class AnthropicGenerator(GeneratorBase):
    def __init__(self, cfg: AnthropicGeneratorConfig) -> None:
        from anthropic import Anthropic

        self.client = Anthropic(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
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
                responses[-1].append(response.content)
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

    def score(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("API based model does not support scoring")

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
