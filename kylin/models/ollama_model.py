import logging

from ollama import Client
from transformers import GenerationConfig


logger = logging.getLogger("OllamaGenerator")


class OllamaGenerator:
    def __init__(
        self,
        model_name: str,
        base_url: str = None,
        verbose: bool = False,
    ) -> None:
        self.client = Client(
            host=base_url,
        )
        self.model_name = model_name
        if not verbose:
            logger = logging.getLogger("httpx")
            logger.setLevel(logging.WARNING)
        self._check()
        return

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        responses = []
        options = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "temperature": generation_config.temperature,
            "num_predict": generation_config.max_new_tokens,
        }
        for conv in prompts:
            response = self.client.chat(
                model=self.model_name, messages=conv, options=options
            )
            responses.append(response["message"]["content"])
        return responses

    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        responses = []
        options = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "temperature": generation_config.temperature,
            "num_predict": generation_config.max_new_tokens,
        }
        for prefix in prefixes:
            response = self.client.generate(
                model=self.model_name,
                prompt=prefix,
                raw=True,
                options=options,
            )
            responses.append(response["message"]["content"])
        return responses

    def _check(self) -> None:
        models = [i["name"] for i in self.client.list()["models"]]
        if self.model_name not in models:
            raise ValueError(f"Model {self.model_name} not found in {models}")
        return
