from transformers import GenerationConfig, PreTrainedTokenizer

from .hf_model import HFGenerator
from .model_base import GeneratorBase
from .ollama_model import OllamaGenerator
from .openai_model import OpenAIGenerator
from .vllm_model import VLLMGenerator


def load_generator(
    model_type: str,
    model_path: str = None,
    gpu_memory_utilization: float = 0.85,
    pipeline_parallel: bool = True,
    base_url: str = None,
    api_key: str = "EMPTY",
    model_name: str = None,
) -> GeneratorBase:
    match model_type:
        case "vllm":
            assert model_path is not None
            model = VLLMGenerator(
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        case "hf":
            assert model_path is not None
            model = HFGenerator(
                model_path=model_path,
                pipeline_parallel=pipeline_parallel,
            )
        case "openai":
            assert model_name is not None
            model = OpenAIGenerator(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
            )
        case "ollama":
            assert model_name is not None
            model = OllamaGenerator(
                model_name=model_name,
                base_url=base_url,
            )
        case _:
            raise ValueError(f"Not supported model: {model_type}")
    return model


def load_generation_config(
    max_gen_tokens: int,
    temperature: float = 0.0,
    top_k: int = 1,
    top_p: float = 1.0,
    model: GeneratorBase = None,
) -> GenerationConfig:
    tokenizer: PreTrainedTokenizer = None
    if isinstance(model, VLLMGenerator):
        tokenizer = model.model.get_tokenizer()
    if isinstance(model, HFGenerator):
        tokenizer = model.tokenizer
    if tokenizer is not None:
        stop_ids = [tokenizer.eos_token_id]
        if "llama-3" in tokenizer.name_or_path.lower():
            stop_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    else:
        stop_ids = []
    return GenerationConfig(
        do_sample=True,
        max_new_tokens=max_gen_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        forced_eos_token_id=stop_ids,
    )
