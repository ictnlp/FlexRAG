from functools import partial

from transformers import AutoConfig, PreTrainedTokenizer
from vllm import LLM, SamplingParams

from kylin.utils import apply_template_llama3, apply_template_phi3


def load_vllm(
    model_path: str,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_gen_tokens: int = 512,
) -> tuple[LLM, PreTrainedTokenizer, SamplingParams, callable]:
    model_cfg = AutoConfig.from_pretrained(model_path)
    cfg_name = model_cfg.__class__.__name__
    model = LLM(
        model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=min(model_cfg.max_position_embeddings, 16384),
        trust_remote_code=True,
    )
    tokenizer = model.get_tokenizer()
    stop_ids = [tokenizer.eos_token_id]
    if cfg_name == "LlamaConfig":
        stop_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_gen_tokens,
        stop_token_ids=stop_ids,
    )
    prompt_func = get_prompt_func(
        model=model,
        tokenizer=tokenizer,
    )
    return model, tokenizer, sampling_params, prompt_func


def get_prompt_func(model: LLM, tokenizer: PreTrainedTokenizer) -> callable:
    cfg_name = model.llm_engine.model_config.hf_config.__class__.__name__
    match cfg_name:
        case "Phi3Config":
            return partial(apply_template_phi3, tokenizer=tokenizer)
        case "Phi3SmallConfig":
            return partial(apply_template_phi3, tokenizer=tokenizer)
        case "LlamaConfig":
            return partial(apply_template_llama3, tokenizer=tokenizer)
        case _:
            raise ValueError(f"Unsupported config: {cfg_name}")
    return
