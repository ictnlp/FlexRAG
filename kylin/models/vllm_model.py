from dataclasses import dataclass

from omegaconf import MISSING
from transformers import AutoConfig
from vllm import LLM, SamplingParams

from kylin.prompt import load_template, ChatPrompt
from kylin.utils import Choices

from .model_base import Generators, GenerationConfig, GeneratorBase, GeneratorBaseConfig


@dataclass
class VLLMGeneratorConfig(GeneratorBaseConfig):
    model_path: str = MISSING
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 16384
    tensor_parallel: int = 1
    load_dtype: Choices(["auto", "float32", "float16", "bfloat16"]) = "auto"  # type: ignore


@Generators("vllm", config_class=VLLMGeneratorConfig)
class VLLMGenerator(GeneratorBase):
    def __init__(self, cfg: VLLMGeneratorConfig) -> None:
        model_cfg = AutoConfig.from_pretrained(cfg.model_path)
        cfg_name = model_cfg.__class__.__name__
        self.model = LLM(
            cfg.model_path,
            dtype=str(cfg.load_dtype),
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            tensor_parallel_size=cfg.tensor_parallel,
            max_model_len=min(model_cfg.max_position_embeddings, cfg.max_model_len),
            trust_remote_code=True,
        )
        self.tokenizer = self.model.get_tokenizer()
        stop_ids = [self.tokenizer.eos_token_id]
        if cfg_name == "LlamaConfig":
            stop_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.template = load_template(model=self.model, tokenizer=self.tokenizer)
        return

    def generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        if generation_config.eos_token_id is not None:
            stop_token_ids = [generation_config.eos_token_id]
        else:
            stop_token_ids = [self.tokenizer.eos_token_id]
        sampling_params = SamplingParams(
            n=generation_config.sample_num,
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            stop_token_ids=stop_token_ids,
        )
        responses = self.model.generate(
            prompts=prefixes,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        responses = [[i.text for i in resp.outputs] for resp in responses]
        return responses

    def chat(
        self,
        prompts: list[ChatPrompt],
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> list[list[str]]:
        prefixes = [self.template.render_to_text(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)
