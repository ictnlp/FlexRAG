from functools import partial

from transformers import PreTrainedModel, PreTrainedTokenizer
from vllm import LLM


def apply_template_llama3(
    history: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    sys_prompt: str = "You are a pirate chatbot who always responds in pirate speak!",
) -> str:
    # add system prompt
    if len(history) == 0:
        history.append({"role": "system", "content": sys_prompt})
    if history[0]["role"] != "system":
        history.insert(0, {"role": "system", "content": sys_prompt})

    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    return prompt


def apply_template_phi3(
    history: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    sys_prompt: str = None,
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
        case _:
            raise ValueError(f"Unsupported config: {cfg_name}")
    return
