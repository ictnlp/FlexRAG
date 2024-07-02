import logging
from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

from .model_base import (
    GeneratorBase,
    GeneratorConfig,
    GenerationConfig,
)


@dataclass
class OpenAIGeneratorConfig(GeneratorConfig):
    model_name: str = MISSING
    base_url: Optional[str] = None
    api_key: str = "EMPTY"
    verbose: bool = False


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
        return

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        responses = []
        if "llama-3" in self.model_name.lower():
            extra_body = {"stop_token_ids": [128009]}  # hotfix for llama-3
        else:
            extra_body = None
        for conv in prompts:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conv,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_new_tokens,
                top_p=generation_config.top_p,
                extra_body=extra_body,
            )
            responses.append(response.choices[0].message.content)
        return responses

    def generate(
        self, prefixes: list[str], generation_config: GenerationConfig = None
    ) -> list[str]:
        responses = []
        for prefix in prefixes:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prefix,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_new_tokens,
                top_p=generation_config.top_p,
            )
            responses.append(response.choices[0].text)
        return responses

    def _check(self):
        return self.client.status.get()
