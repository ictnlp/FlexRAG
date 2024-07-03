from functools import partial

from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM


def apply_template_general(default_system_prompt: str = None):
    def apply_template_general_(
        history: list[dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        sys_prompt: str = default_system_prompt,
    ) -> str:
        # add system prompt
        if (len(history) == 0) and (sys_prompt is not None):
            history.append({"role": "system", "content": sys_prompt})
        if (history[0]["role"] != "system") and (sys_prompt is not None):
            history.insert(0, {"role": "system", "content": sys_prompt})

        # apply template
        prompt = tokenizer.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True
        )
        return prompt

    return apply_template_general_


apply_template_phi3 = apply_template_general()
apply_template_llama3 = apply_template_general(
    "You are a pirate chatbot who always responds in pirate speak!"
)
apply_template_qwen2 = apply_template_general("You are a helpful assistant.")


def get_prompt_func(
    model: LLM | PreTrainedModel, tokenizer: PreTrainedTokenizer
) -> callable:
    # get model name
    if isinstance(model, LLM):
        cfg_name = model.llm_engine.model_config.hf_config.__class__.__name__
    elif isinstance(model, PreTrainedModel):
        cfg_name = model.config.__class__.__name__

    # decide template function
    match cfg_name:
        case "Phi3Config":
            return partial(apply_template_phi3, tokenizer=tokenizer)
        case "Phi3SmallConfig":
            return partial(apply_template_phi3, tokenizer=tokenizer)
        case "LlamaConfig":
            return partial(apply_template_llama3, tokenizer=tokenizer)
        case "Qwen2Config":
            return partial(apply_template_qwen2, tokenizer=tokenizer)
        case _:
            raise ValueError(f"Unsupported config: {cfg_name}")
    return
