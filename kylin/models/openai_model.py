import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from omegaconf import MISSING

from kylin.prompt import ChatPrompt

from .model_base import (
    EncoderBase,
    EncoderBaseConfig,
    Encoders,
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
    Generators,
)


@dataclass
class OpenAIGeneratorConfig(GeneratorBaseConfig):
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    verbose: bool = False


@dataclass
class OpenAIEncoderConfig(EncoderBaseConfig):
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    verbose: bool = False
    dimension: int = 512


@Generators("openai", config_class=OpenAIGeneratorConfig)
class OpenAIGenerator(GeneratorBase):
    def __init__(self, cfg: OpenAIGeneratorConfig) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
        self.model_name = cfg.model_name
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses = []
        gen_cfg = self.prepare_generation_config(generation_config)
        for prompt in prompts:
            prompt = prompt.to_list()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                **gen_cfg,
            )
            responses.append([i.message.content for i in response.choices])
        return responses

    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses = []
        gen_cfg = self.prepare_generation_config(generation_config)
        for prefix in prefixes:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prefix,
                **gen_cfg,
            )
            responses.append([i.message.content for i in response.choices])
        return responses

    def prepare_generation_config(self, generation_config: GenerationConfig) -> dict:
        if "llama-3" in self.model_name.lower():
            extra_body = {"stop_token_ids": [128009]}  # hotfix for llama-3
        else:
            extra_body = None
        return {
            "temperature": generation_config.temperature,
            "max_tokens": generation_config.max_new_tokens,
            "top_p": generation_config.top_p,
            "n": generation_config.sample_num,
            "extra_body": extra_body,
        }

    def _check(self):
        model_lists = [i.id for i in self.client.models.list().data]
        assert self.model_name in model_lists, f"Model {self.model_name} not found"


@Encoders("openai", config_class=OpenAIEncoderConfig)
class OpenAIEncoder(EncoderBase):
    def __init__(self, cfg: OpenAIEncoderConfig) -> None:
        from openai import OpenAI

        self.client = OpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
        )
        self.model_name = cfg.model_name
        self.dimension = cfg.dimension
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for text in texts:
            text = text.replace("\n", " ")
            embeddings.append(
                self.client.embeddings.create(model=self.model_name, text=text)
                .data[0]
                .embedding
            )[: self.dimension]
        return np.array(embeddings)

    @property
    def embedding_size(self):
        return self.dimension

    def _check(self):
        model_lists = [i.id for i in self.client.models.list().data]
        assert self.model_name in model_lists, f"Model {self.model_name} not found"
