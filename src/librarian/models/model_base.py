from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL.Image import Image

from librarian.prompt import ChatPrompt, MultiModelChatPrompt
from librarian.utils import Register, LOGGER_MANAGER


logger = LOGGER_MANAGER.get_logger("librarian.models")


@dataclass
class GeneratorBaseConfig: ...


@dataclass
class GenerationConfig:
    do_sample: bool = True
    sample_num: int = 1
    temperature: float = 1.0
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 50
    eos_token_id: Optional[int] = None
    stop_str: Optional[str] = None

    def __post_init__(self):
        # check values
        assert self.sample_num > 0, "sample_num must be greater than 0"
        if self.sample_num > 1:
            assert self.do_sample, "do_sample must be True when sample_num > 1"
        assert self.temperature >= 0, "temperature must be greater than or equal to 0"
        assert self.max_new_tokens > 0, "max_new_tokens must be greater than 0"
        assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1"
        assert self.top_k > 0, "top_k must be greater than 0"


class GeneratorBase(ABC):
    @abstractmethod
    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """chat with the model using model templates.

        Args:
            prompts (list[ChatPrompt]): A batch of ChatPrompts.
            generation_config (GenerationConfig, optional): GenerationConfig. Defaults to None.

        Returns:
            list[list[str]]: A batch of chat responses.
        """
        return

    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """The async version of chat."""
        logger.warning(
            "Current encoder does not support asyncronous chat, thus the code will be run in syncronous mode"
        )
        return self.chat(prompts=prompts, generation_config=generation_config)

    @abstractmethod
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """generate text with the model using the given prefixes.

        Args:
            prefixes (list[str]): A batch of prefixes.
            generation_config (GenerationConfig, optional): GenerationConfig. Defaults to None.

        Returns:
            list[list[str]]: A batch of generated text.
        """
        return

    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """The async version of generate."""
        logger.warning(
            "Current generator does not support asyncronous generate, thus the code will be run in syncronous mode"
        )
        return self.generate(prefixes=prefixes, generation_config=generation_config)


class VLMGeneratorBase(GeneratorBase):
    @abstractmethod
    def chat(
        self,
        prompts: list[MultiModelChatPrompt],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """chat with the model using model templates.

        Args:
            prompts (list[MultiModelChatPrompt]): A batch of MultiModelChatPrompts.
            generation_config (GenerationConfig, optional): GenerationConfig. Defaults to None.

        Returns:
            list[list[str]]: A batch of chat responses.
        """
        return

    @abstractmethod
    def generate(
        self,
        prefixes: list[str],
        images: list[Image],
        generation_config: GenerationConfig = None,
    ) -> list[list[str]]:
        """generate text with the model using the given prefixes.

        Args:
            prefixes (list[str]): A batch of prefixes.
            images (list[Image]): A batch of images.
            generation_config (GenerationConfig, optional): GenerationConfig. Defaults to None.

        Returns:
            list[list[str]]: A batch of generated text.
        """
        return


@dataclass
class EncoderBaseConfig: ...


class EncoderBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """encode the given texts into embeddings.

        Args:
            texts (list[str]): A batch of texts.

        Returns:
            np.ndarray: A batch of embeddings.
        """
        return

    async def async_encode(self, texts: list[str]) -> np.ndarray:
        """The async version of encode."""
        logger.warning(
            "Current encoder does not support asyncronous encode, thus the code will be run in syncronous mode"
        )
        return self.encode(texts)

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        return


GENERATORS = Register[GeneratorBase]("generator")
ENCODERS = Register[EncoderBase]("encoder")
