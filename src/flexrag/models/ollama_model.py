import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
from numpy import ndarray

from flexrag.prompt import ChatPrompt
from flexrag.utils import LOGGER_MANAGER, TIME_METER, configure

from .model_base import (
    ENCODERS,
    GENERATORS,
    EncoderBase,
    EncoderBaseConfig,
    GenerationConfig,
    GeneratorBase,
)

logger = LOGGER_MANAGER.get_logger("flexrag.models.ollama")


@configure
class OllamaGeneratorConfig:
    """Configuration for the OllamaGenerator.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server.
        Default is 'http://localhost:11434/'.
    :type base_url: str
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param num_ctx: The number of context tokens to use. Default is 4096.
    :type num_ctx: int
    :param max_concurrency: The maximum number of concurrent generation requests. Default is 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    base_url: str = "http://localhost:11434/"
    verbose: bool = False
    num_ctx: int = 4096
    max_concurrency: int = 1


@GENERATORS("ollama", config_class=OllamaGeneratorConfig)
class OllamaGenerator(GeneratorBase):
    def __init__(self, cfg: OllamaGeneratorConfig) -> None:
        from ollama import AsyncClient, Client

        # initialize ollama client
        self.client = Client(host=cfg.base_url)
        self.async_client = AsyncClient(host=cfg.base_url)

        # set arguments
        assert cfg.model_name is not None, "`model_name` must be provided"
        self.model_name = cfg.model_name
        self.max_length = cfg.num_ctx
        self.max_concurrency = cfg.max_concurrency

        # prepare logger
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TIME_METER("generator.ollama_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as ollama does not support sample_num, we sample multiple times
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        options = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        def _create(prompt: ChatPrompt) -> str:
            return self.client.chat(
                model=self.model_name,
                messages=prompt.to_list(),
                options=options,
            ).message.content

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

    @TIME_METER("generator.ollama_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as ollama does not support sample_num, we sample multiple times
        prompts = [prompts] if not isinstance(prompts, list) else prompts
        options = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(prompt: ChatPrompt) -> str:
            async with semaphore:
                return await self.async_client.chat(
                    model=self.model_name,
                    messages=prompt.to_list(),
                    options=options,
                ).message.content

        tasks = prompts * sample_num
        tasks = asyncio.create_task(_create(p) for p in tasks)
        responses = [await task for task in tasks]
        responses = [
            responses[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prompts))
        ]
        return responses

    @TIME_METER("generator.ollama_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        # as ollama does not support sample_num, we sample multiple times
        prefixes = [prefixes] if not isinstance(prefixes, list) else prefixes
        options = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        def _create(prefix: str) -> str:
            return self.client.generate(
                model=self.model_name,
                prompt=prefix,
                raw=True,
                options=options,
            ).response

        tasks = prefixes * sample_num
        if self.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                responses = list(pool.map(_create, tasks))
        else:
            responses = []
            for task in tasks:
                responses.append(_create(task))
        responses: list[list[str]] = [
            responses[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prefixes))
        ]
        return responses

    @TIME_METER("generator.ollama_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [prefixes] if not isinstance(prefixes, list) else prefixes
        options = self._get_options(generation_config)
        sample_num = generation_config.sample_num

        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(prefix: str) -> str:
            async with semaphore:
                return await self.async_client.generate(
                    model=self.model_name,
                    prompt=prefix,
                    raw=True,
                    options=options,
                ).response

        tasks = prefixes * sample_num
        tasks = asyncio.create_task(_create(p) for p in tasks)
        responses = [await task for task in tasks]
        responses = [
            responses[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prefixes))
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


@configure
class OllamaEncoderConfig(EncoderBaseConfig):
    """Configuration for the OllamaEncoder.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server.
        Default is 'http://localhost:11434/'.
    :type base_url: str
    :param prompt: The prompt to use. Default is None.
    :type prompt: Optional[str]
    :param verbose: Whether to show verbose logs. Default is False.
    :type verbose: bool
    :param embedding_size: The size of the embeddings. Default is 768.
    :type embedding_size: int
    :param max_concurrency: The maximum number of concurrent encoding requests. Default is 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    base_url: str = "http://localhost:11434/"
    prompt: Optional[str] = None
    verbose: bool = False
    embedding_size: int = 768
    max_concurrency: int = 1


@ENCODERS("ollama", config_class=OllamaEncoderConfig)
class OllamaEncoder(EncoderBase):
    def __init__(self, cfg: OllamaEncoderConfig) -> None:
        super().__init__(cfg)
        from ollama import AsyncClient, Client

        # initialize ollama client
        self.client = Client(host=cfg.base_url)
        self.async_client = AsyncClient(host=cfg.base_url)

        # set arguments
        assert cfg.model_name is not None, "`model_name` must be provided"
        self.model_name = cfg.model_name
        self.prompt = cfg.prompt
        self._embedding_size = cfg.embedding_size
        self.max_concurrency = cfg.max_concurrency

        # set logger
        if not cfg.verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    @TIME_METER("encoder.ollama_encode")
    def _encode(self, texts: list[str]) -> ndarray:
        texts = [texts] if not isinstance(texts, list) else texts
        if self.prompt:
            texts = [f"{self.prompt} {text}" for text in texts]

        def _create(text: str) -> list[float]:
            embed = self.client.embeddings(model=self.model_name, prompt=text)
            return embed["embedding"]

        if self.max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
                embeddings = pool.map(_create, texts)
                embeddings = list(embeddings)
        else:
            embeddings = []
            for text in texts:
                embeddings.append(_create(text))
        embeddings = np.array(embeddings)
        return embeddings[:, : self.embedding_size]

    @TIME_METER("encoder.ollama_encode")
    async def async_encode(self, texts: list[str]) -> ndarray:
        texts = [texts] if not isinstance(texts, list) else texts
        if self.prompt:
            texts = [f"{self.prompt} {text}" for text in texts]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _create(text: str) -> list[float]:
            async with semaphore:
                return await self.async_client.embeddings(
                    model=self.model_name,
                    prompt=text,
                )["embedding"]

        tasks = [asyncio.create_task(_create(text)) for text in texts]
        embeddings = np.array([await task for task in tasks])
        return embeddings[:, : self.embedding_size]

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def _check(self) -> None:
        models = [i["name"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return
