from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GeneratorConfig: ...


@dataclass
class EncoderConfig: ...


@dataclass
class GenerationConfig:
    do_sample: bool = True
    temperature: float = 1.0
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 0
    eos_token_id: Optional[int] = None


class GeneratorBase(ABC):
    @abstractmethod
    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        return

    @abstractmethod
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        return


class EncoderBase(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        return

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        return
