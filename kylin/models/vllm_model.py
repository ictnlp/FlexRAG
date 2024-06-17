import torch
from transformers import AutoConfig, GenerationConfig
from vllm import LLM, SamplingParams

from .model_base import GeneratorBase
from .utils import get_prompt_func


class VLLMGenerator(GeneratorBase):
    def __init__(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        model_cfg = AutoConfig.from_pretrained(model_path)
        cfg_name = model_cfg.__class__.__name__
        self.model = LLM(
            model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=torch.cuda.device_count(),
            max_model_len=min(model_cfg.max_position_embeddings, 16384),
            trust_remote_code=True,
        )
        self.tokenizer = self.model.get_tokenizer()
        stop_ids = [self.tokenizer.eos_token_id]
        if cfg_name == "LlamaConfig":
            stop_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.prompt_func = get_prompt_func(
            model=self.model,
            tokenizer=self.tokenizer,
        )
        return

    def generate(
        self, prefixes: list[str], generation_config: GenerationConfig = None
    ) -> list[str]:
        sampling_params = SamplingParams(
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_k=generation_config.top_k,
            top_p=generation_config.top_p,
            stop_token_ids=generation_config.forced_eos_token_id,
        )
        responses = self.model.generate(
            prompts=prefixes,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        responses = [i.outputs[0].text for i in responses]
        return responses

    def chat(
        self,
        prompts: list[list[dict[str, str]]],
        generation_config: GenerationConfig = None,
    ) -> list[str]:
        prefixes = [self.prompt_func(prompt) for prompt in prompts]
        return self.generate(prefixes, generation_config)
