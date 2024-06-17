from abc import ABC, abstractmethod
from argparse import ArgumentParser

from transformers import GenerationConfig


class GeneratorBase(ABC):
    @abstractmethod
    def chat(
        self, prompts: list[list[dict[str, str]]], generation_config: GenerationConfig = None
    ) -> list[str]:
        return

    @abstractmethod
    def generate(
        self, prefixes: list[str], generation_config: GenerationConfig = None
    ) -> list[str]:
        return


class EncoderBase(ABC):
    def encode(self, texts: list[str]): ...
