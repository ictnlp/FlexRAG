from transformers import PretrainedConfig

from librarian.utils import LOGGER_MANAGER


logger = LOGGER_MANAGER.get_logger("librarian.models.utils")


def guess_model_name(model_cfg: PretrainedConfig) -> str | None:
    arch_name = getattr(model_cfg, "architectures", [None])[0]
    hidden_size = getattr(model_cfg, "hidden_size", None)
    max_length = getattr(model_cfg, "max_position_embeddings", None)
    eos_token_id = getattr(model_cfg, "eos_token_id", None)
    vocab_size = getattr(model_cfg, "vocab_size", None)
    name_or_path = getattr(model_cfg, "_name_or_path", None)

    # Qwen-2 series
    if arch_name == "Qwen2ForCausalLM":
        if hidden_size == 3584:
            if eos_token_id == 151645:
                return "Qwen/Qwen2-7B-Instruct"
            elif eos_token_id == 151643:
                return "Qwen/Qwen2-7B"
        elif hidden_size == 8192:
            if eos_token_id == 151645:
                return "Qwen/Qwen2-72B-Instruct"
            elif eos_token_id == 151643:
                return "Qwen/Qwen2-72B"

    # Llama-3/Llama-3.1 series
    if (arch_name == "LlamaForCausalLM") and (vocab_size == 128256):
        if max_length == 8192:
            if hidden_size == 4096:
                if eos_token_id == 128001:
                    return "meta-llama/Meta-Llama-3-8B"
                elif eos_token_id == 128009:
                    return "meta-llama/Meta-Llama-3-8B-Instruct"
            elif hidden_size == 8192:
                if eos_token_id == 128001:
                    return "meta-llama/Meta-Llama-3-70B"
                elif eos_token_id == 128009:
                    return "meta-llama/Meta-Llama-3-70B-Instruct"
        elif max_length == 131072:
            if hidden_size == 4096:
                if eos_token_id == 128001:
                    return "meta-llama/Meta-Llama-3.1-8B"
                elif eos_token_id == [128001, 128008, 128009]:
                    return "meta-llama/Meta-Llama-3.1-8B-Instruct"
            elif hidden_size == 8192:
                if eos_token_id == 128001:
                    return "meta-llama/Meta-Llama-3.1-70B"
                elif eos_token_id == [128001, 128008, 128009]:
                    return "meta-llama/Meta-Llama-3.1-70B-Instruct"

    # Phi-3/Phi-3.5 series
    if arch_name == "Phi3ForCausalLM":
        if "Phi-3.5" in name_or_path:
            return "microsoft/Phi-3.5-mini-instruct"
        if hidden_size == 3072:
            if max_length == 4096:
                return "microsoft/Phi-3-mini-4k-instruct"
            elif max_length == 131072:
                return "microsoft/Phi-3-mini-128k-instruct"
        elif hidden_size == 5120:
            if max_length == 4096:
                return "microsoft/Phi-3-medium-4k-instruct"
            elif max_length == 131072:
                return "microsoft/Phi-3-medium-128k-instruct"
    elif arch_name == "Phi3SmallForCausalLM":
        if max_length == 8192:
            return "microsoft/Phi-3-small-8k-instruct"
        elif max_length == 131072:
            return "microsoft/Phi-3-small-128k-instruct"
    elif arch_name == "Phi-3.5-MoE-instruct":
        return "microsoft/Phi-3.5-MoE-instruct"

    logger.warning(f"Unable to guess model name from config: {model_cfg}")
    return None
