import asyncio
from dataclasses import dataclass

from omegaconf import MISSING

from flexrag.prompt import ChatPrompt
from flexrag.utils import TIME_METER

from .model_base import GENERATORS, GenerationConfig, GeneratorBase


@dataclass
class LlamacppGeneratorConfig:
    """Configuration for LlamacppGenerator.

    :param model_path: The path to the model.
    :type model_path: str
    :param use_gpu: Whether to use GPU. Default is False.
    :type use_gpu: bool
    :param verbose: Whether to output verbose logs. Default is False.
    :type verbose: bool
    """

    model_path: str = MISSING
    use_gpu: bool = False
    verbose: bool = False


@GENERATORS("llamacpp", config_class=LlamacppGeneratorConfig)
class LlamacppGenerator(GeneratorBase):
    def __init__(self, cfg: LlamacppGeneratorConfig) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=cfg.model_path,
            n_gpu_layers=-1 if cfg.use_gpu else 0,
            use_mmap=True,
            verbose=cfg.verbose,
        )
        return

    @TIME_METER("llamacpp_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        options, sample_num = self._get_options(generation_config)
        for prefix in prefixes:
            # as llamacpp does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(sample_num):
                response = self.model.create_completion(
                    prompt=prefix,
                    **options,
                )
                responses[-1].append(response["choices"][0]["text"])
        return responses

    @TIME_METER("llamacpp_generate")
    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        responses: list[list[str]] = []
        options, sample_num = self._get_options(generation_config)
        for prefix in prefixes:
            responses.append([])
            for _ in range(sample_num):
                r = await asyncio.to_thread(
                    self.model.create_completion,
                    prompt=prefix,
                    **options,
                )
                responses[-1].append(r["choices"][0]["text"])
        return responses

    @TIME_METER("llamacpp_generate")
    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        options, sample_num = self._get_options(generation_config)
        for prompt in prompts:
            # as llamacpp does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(sample_num):
                response = self.model.create_chat_completion(
                    messages=prompt.to_list(),
                    **options,
                )
                responses[-1].append(response["choices"][0]["message"]["content"])
        return responses

    @TIME_METER("llamacpp_generate")
    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prompts, list):
            prompts = [prompts]
        responses: list[list] = []
        options, sample_num = self._get_options(generation_config)
        for prompt in prompts:
            responses.append([])
            for _ in range(sample_num):
                r = await asyncio.to_thread(
                    self.model.create_chat_completion,
                    messages=prompt.to_list(),
                    **options,
                )
                responses[-1].append(r["choices"][0]["message"]["content"])
        return responses

    def _get_options(self, generation_config: GenerationConfig) -> tuple[dict, int]:
        if generation_config is None:
            generation_config = GenerationConfig()
        return {
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "max_tokens": generation_config.max_new_tokens,
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "stop": list(generation_config.stop_str),
        }, generation_config.sample_num
