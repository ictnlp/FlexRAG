import asyncio
from dataclasses import dataclass

from omegaconf import MISSING
from transformers import AutoConfig, PretrainedConfig

from flexrag.prompt import load_template, ChatPrompt
from flexrag.utils import Choices, TIME_METER, LOGGER_MANAGER

from .model_base import GENERATORS, GenerationConfig, GeneratorBase
from .utils import guess_model_name

logger = LOGGER_MANAGER.get_logger("flexrag.models.vllm")


@dataclass
class VLLMGeneratorConfig:
    """Configuration for VLLMGenerator.

    :param model_path: Path to the model. Required.
    :type model_path: str
    :param gpu_memory_utilization: Fraction of GPU memory to use. Default to 0.85.
    :type gpu_memory_utilization: float
    :param max_model_len: Maximum length of the model. Defaults to 16384.
    :type max_model_len: int
    :param tensor_parallel: The number of tensor parallel. Defaults to 1.
    :type tensor_parallel: int
    :param load_dtype: The dtype to load the model. Defaults to "auto". Available options are "auto", "float32", "float16", "bfloat16".
    :type load_dtype: str
    :param use_minference: Whether to use minference for Long Sequence Inference. Defaults to False.
    :type use_minference: bool
    """

    model_path: str = MISSING
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 16384
    tensor_parallel: int = 1
    load_dtype: Choices(["auto", "float32", "float16", "bfloat16"]) = "auto"  # type: ignore
    use_minference: bool = False
    trust_remote_code: bool = False


@GENERATORS("vllm", config_class=VLLMGeneratorConfig)
class VLLMGenerator(GeneratorBase):
    def __init__(self, cfg: VLLMGeneratorConfig) -> None:
        from vllm import LLM

        # try to load model arguments from model config
        model_cfg: PretrainedConfig = AutoConfig.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )
        model_name = guess_model_name(model_cfg)
        max_length = min(
            getattr(model_cfg, "max_position_embeddings", cfg.max_model_len),
            cfg.max_model_len,
        )

        # load model
        self.model = LLM(
            cfg.model_path,
            dtype=str(cfg.load_dtype),
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel,
            max_model_len=max_length,
            trust_remote_code=cfg.trust_remote_code,
            enforce_eager=True if cfg.use_minference else False,
        )
        self.tokenizer = self.model.get_tokenizer()
        self.template = load_template(model_name=model_name, tokenizer=self.tokenizer)

        # load minference
        if cfg.use_minference:
            try:
                from minference import MInference

                inf_patch = MInference("vllm", model_name)
                self.model = inf_patch(self.model)
            except Exception as e:
                logger.warning(f"Unable to load minference: {e}")
                logger.warning("Fallback to normal mode.")
        return

    @TIME_METER("vllm_generate")
    def _generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        responses = self.model.generate(
            prefixes,
            sampling_params=self._get_options(generation_config),
            use_tqdm=False,
        )
        responses = [[i.text for i in resp.outputs] for resp in responses]
        return responses

    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        responses = await asyncio.to_thread(
            self.model.generate,
            prefixes,
            sampling_params=self._get_options(generation_config),
            use_tqdm=False,
        )
        responses = [[i.text for i in resp.outputs] for resp in responses]
        return responses

    def _chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [self.template.render_to_text(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)

    async def async_chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [self.template.render_to_text(prompt) for prompt in prompts]
        return await self.async_generate(prefixes, generation_config)

    def _get_options(self, generation_config: GenerationConfig):
        from vllm import SamplingParams

        if generation_config.eos_token_id is not None:
            stop_token_ids = [generation_config.eos_token_id]
        else:
            stop_token_ids = [self.tokenizer.eos_token_id]
        return SamplingParams(
            n=generation_config.sample_num,
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            stop_token_ids=stop_token_ids,
            stop=generation_config.stop_str,
        )
