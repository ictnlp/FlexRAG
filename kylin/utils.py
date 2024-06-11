from transformers import PreTrainedTokenizer


def apply_template_llama3(
    history: list[dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    sys_prompt: str = "You are a pirate chatbot who always responds in pirate speak!",
) -> str:
    # add system prompt
    if len(history) == 0:
        history.append({"role": "system", "content": sys_prompt})
    if history[0]["role"] != history:
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
    if (history[0]["role"] != history) and (sys_prompt is not None):
        history.insert(0, {"role": "system", "content": sys_prompt})

    # apply template
    prompt = tokenizer.apply_chat_template(
        history, tokenize=False, add_generation_prompt=True
    )
    return prompt

