import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from kylin.prompt import ChatPrompt
from kylin.utils import TimeMeter

from .model_base import (
    GenerationConfig,
    GeneratorBase,
    GeneratorBaseConfig,
    Generators,
    EncoderBase,
    EncoderBaseConfig,
    Encoders,
)

logger = logging.getLogger("OllamaGenerator")


@dataclass
class OllamaGeneratorConfig(GeneratorBaseConfig):
    model_name: str = MISSING
    base_url: str = MISSING
    verbose: bool = False
    num_ctx: int = 4096


@Generators("ollama", config_class=OllamaGeneratorConfig)
class OllamaGenerator(GeneratorBase):
    def __init__(self, cfg: OllamaGeneratorConfig) -> None:
        from ollama import Client

        self.client = Client(host=cfg.base_url)
        self.model_name = cfg.model_name
        self.max_length = cfg.num_ctx
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TimeMeter("ollama_generate")
    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        options = self._get_options(generation_config)
        for prompt in prompts:
            # as ollama does not support sample_num, we sample multiple times
            prompt = prompt.to_list()
            responses.append([])
            for _ in range(generation_config.sample_num):
                response = self.client.chat(
                    model=self.model_name,
                    messages=prompt,
                    options=options,
                )
                responses[-1].append(response["message"]["content"])
        return responses

    @TimeMeter("ollama_generate")
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        options = self._get_options(generation_config)
        for prefix in prefixes:
            # as ollama does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(generation_config.sample_num):
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prefix,
                    raw=True,
                    options=options,
                )
                responses[-1].append(response["message"]["content"])
        return responses

    def _get_options(self, generation_config: GenerationConfig) -> dict:
        return {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "num_predict": generation_config.max_new_tokens,
            "num_ctx": self.max_length,
        }

    def _check(self) -> None:
        models = [i["name"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return


@dataclass
class OllamaEncoderConfig(EncoderBaseConfig):
    model_name: str = MISSING
    base_url: str = MISSING
    prompt: Optional[str] = None
    verbose: bool = False
    embedding_size: int = 768


@Encoders("ollama", config_class=OllamaEncoderConfig)
class OllamaEncoder(EncoderBase):
    def __init__(self, cfg: OllamaEncoderConfig) -> None:
        super().__init__()
        from ollama import Client

        self.client = Client(host=cfg.base_url)
        self.model_name = cfg.model_name
        self.prompt = cfg.prompt
        self._embedding_size = cfg.embedding_size
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TimeMeter("ollama_encode")
    def encode(self, texts: list[str]) -> ndarray:
        if self.prompt:
            texts = [f"{self.prompt} {text}" for text in texts]
        embeddings = []
        for text in texts:
            embeddings.append(
                self.client.embeddings(model=self.model_name, prompt=text)["embedding"]
            )
        embeddings = np.array(embeddings)
        return embeddings[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def _check(self) -> None:
        models = [i["name"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return
