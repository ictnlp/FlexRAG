from dataclasses import dataclass

from omegaconf import MISSING

from kylin.prompt import ChatPrompt
from kylin.utils import TimeMeter

from .model_base import Generators, GenerationConfig, GeneratorBase, GeneratorBaseConfig


@dataclass
class LlamacppGeneratorConfig(GeneratorBaseConfig):
    model_path: str = MISSING
    use_gpu: bool = False


@Generators("llamacpp", config_class=LlamacppGeneratorConfig)
class LlamacppGenerator(GeneratorBase):
    def __init__(self, cfg: LlamacppGeneratorConfig) -> None:
        from llama_cpp import Llama

        self.model = Llama(
            model_path=cfg.model_path,
            n_gpu_layers=-1 if cfg.use_gpu else 0,
            use_mmap=True,
        )
        return

    @TimeMeter("llamacpp_generate")
    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        options = self._get_options(generation_config)
        for prefix in prefixes:
            # as llamacpp does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(generation_config.sample_num):
                response = self.model.create_completion(
                    prompt=prefix,
                    **options,
                )
                responses[-1].append(response["choices"][0]["text"])
        return responses

    @TimeMeter("llamacpp_generate")
    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        responses: list[list[str]] = []
        for prompt in prompts:
            # as llamacpp does not support sample_num, we sample multiple times
            responses.append([])
            for _ in range(generation_config.sample_num):
                response = self.model.create_chat_completion(
                    messages=prompt.to_list(),
                    **self._get_options(generation_config),
                )
                responses[-1].append(response["choices"][0]["message"]["content"])
        return responses

    def score(self, texts: list[str]) -> list[float]:
        raise NotImplementedError("llamacpp does not support scoring")

    def _get_options(self, generation_config: GenerationConfig) -> dict:
        return {
            "temperature": (
                generation_config.temperature if generation_config.do_sample else 0.0
            ),
            "max_tokens": generation_config.max_new_tokens,
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
        }
