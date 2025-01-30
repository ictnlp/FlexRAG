import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from numpy import ndarray
from omegaconf import MISSING

from flexrag.prompt import ChatPrompt
from flexrag.utils import TIME_METER, LOGGER_MANAGER

from .model_base import (
    GenerationConfig,
    GeneratorBase,
    GENERATORS,
    EncoderBase,
    ENCODERS,
)

logger = LOGGER_MANAGER.get_logger("flexrag.models.ollama")


@dataclass
class OllamaGeneratorConfig:
    """Configuration for the OllamaGenerator.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server. Required.
    :type base_url: str
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param num_ctx: The number of context tokens to use. Default is 4096.
    :type num_ctx: int
    :param allow_parallel: Whether to allow parallel generation. Default is True.
    :type allow_parallel: bool
    """

    model_name: str = MISSING
    base_url: str = MISSING
    verbose: bool = False
    num_ctx: int = 4096
    allow_parallel: bool = True


@GENERATORS("ollama", config_class=OllamaGeneratorConfig)
class OllamaGenerator(GeneratorBase):
    def __init__(self, cfg: OllamaGeneratorConfig) -> None:
        from ollama import Client

        self.client = Client(host=cfg.base_url)
        self.model_name = cfg.model_name
        self.max_length = cfg.num_ctx
        self.allow_parallel = cfg.allow_parallel
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TIME_METER("ollama_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as ollama does not support sample_num, we sample multiple times
        options = self._get_options(generation_config)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prompt: [
                        self.client.chat(
                            model=self.model_name,
                            messages=prompt.to_list(),
                            options=options,
                        ).message.content
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
                    response = self.client.chat(
                        model=self.model_name,
                        messages=prompt,
                        options=options,
                    )
                    responses[-1].append(response.message.content)
        return responses

    @TIME_METER("ollama_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        tasks = []
        options = self._get_options(generation_config)
        for prompt in prompts:
            # as ollama does not support sample_num, we sample multiple times
            prompt = prompt.to_list()
            tasks.append([])
            for _ in range(generation_config.sample_num):
                tasks[-1].append(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self.client.chat,
                            model=self.model_name,
                            messages=prompt,
                            options=options,
                        )
                    )
                )
        responses = [
            [(await task).message.content for task in task_list] for task_list in tasks
        ]
        return responses

    @TIME_METER("ollama_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as ollama does not support sample_num, we sample multiple times
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        options = self._get_options(generation_config)
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                responses = pool.map(
                    lambda prefix: [
                        self.client.generate(
                            model=self.model_name,
                            prompt=prefix,
                            raw=True,
                            options=options,
                        ).response
                        for _ in range(generation_config.sample_num)
                    ],
                    prefixes,
                )
                responses = list(responses)
        else:
            responses: list[list[str]] = []
            for prefix in prefixes:
                responses.append([])
                for _ in range(generation_config.sample_num):
                    response = self.client.generate(
                        model=self.model_name,
                        prompt=prefix,
                        raw=True,
                        options=options,
                    )
                    responses[-1].append(response.response)
        return responses

    @TIME_METER("ollama_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        tasks = []
        options = self._get_options(generation_config)
        for prefix in prefixes:
            # as ollama does not support sample_num, we sample multiple times
            tasks.append([])
            for _ in range(generation_config.sample_num):
                tasks[-1].append(
                    asyncio.create_task(
                        asyncio.to_thread(
                            self.client.generate,
                            model=self.model_name,
                            prompt=prefix,
                            raw=True,
                            options=options,
                        )
                    )
                )
        responses = [
            [(await task).response for task in task_list] for task_list in tasks
        ]
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
            "stop": list(generation_config.stop_str),
        }

    def _check(self) -> None:
        models = [i["model"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return


@dataclass
class OllamaEncoderConfig:
    """Configuration for the OllamaEncoder.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server. Required.
    :type base_url: str
    :param prompt: The prompt to use. Default is None.
    :type prompt: Optional[str]
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param embedding_size: The size of the embeddings. Default is 768.
    :type embedding_size: int
    :param allow_parallel: Whether to allow parallel generation. Default is True.
    """

    model_name: str = MISSING
    base_url: str = MISSING
    prompt: Optional[str] = None
    verbose: bool = False
    embedding_size: int = 768
    allow_parallel: bool = True


@ENCODERS("ollama", config_class=OllamaEncoderConfig)
class OllamaEncoder(EncoderBase):
    def __init__(self, cfg: OllamaEncoderConfig) -> None:
        super().__init__()
        from ollama import Client

        self.client = Client(host=cfg.base_url)
        self.model_name = cfg.model_name
        self.prompt = cfg.prompt
        self._embedding_size = cfg.embedding_size
        self.allow_parallel = cfg.allow_parallel
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TIME_METER("ollama_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        if self.prompt:
            texts = [f"{self.prompt} {text}" for text in texts]
        if self.allow_parallel:
            with ThreadPoolExecutor() as pool:
                embeddings = pool.map(
                    lambda text: self.client.embeddings(
                        model=self.model_name, prompt=text
                    )["embedding"],
                    texts,
                )
                embeddings = list(embeddings)
        else:
            embeddings = []
            for text in texts:
                embeddings.append(
                    self.client.embeddings(model=self.model_name, prompt=text)[
                        "embedding"
                    ]
                )
        embeddings = np.array(embeddings)
        return embeddings[:, : self.embedding_size]

    @TIME_METER("ollama_encode")
    async def async_encode(self, texts: list[str]) -> ndarray:
        if self.prompt:
            texts = [f"{self.prompt} {text}" for text in texts]
        tasks = []
        for text in texts:
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(
                        self.client.embeddings,
                        model=self.model_name,
                        prompt=text,
                    )
                )
            )
        embeddings = np.array([(await task)["embedding"] for task in tasks])
        return embeddings[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def _check(self) -> None:
        models = [i["name"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return
