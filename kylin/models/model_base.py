from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from kylin.prompt import ChatPrompt
from kylin.utils import Register


Encoders = Register("Encoders")
Generators = Register("Generators")
Rankers = Register("Rankers")


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


@dataclass
class EncoderBaseConfig: ...


class EncoderBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        return


@dataclass
class RankerConfig: ...


@dataclass
class RankingResult:
    query: str
    candidates: list[str]
    scores: Optional[list[float]] = None
    ranking: Optional[list[int]] = None


class RankerBase(ABC):
    @abstractmethod
    def rank(
        self,
        query: str,
        candidates: list[str],
    ) -> RankingResult:
        return
