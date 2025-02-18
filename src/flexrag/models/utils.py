import transformers
import torch
import torch.distributed as dist
from transformers import AutoConfig, PretrainedConfig

from flexrag.utils import LOGGER_MANAGER


logger = LOGGER_MANAGER.get_logger("flexrag.models.utils")


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


def get_gpu_capability(device_id: list[int]) -> float:
    """Get the GPU capability of the first GPU."""
    if len(device_id) == 0:
        return 0.0
    try:
        caps = []
        for device in device_id:
            cap = torch.cuda.get_device_capability(device)
            caps.append(float(f"{cap[0]}.{cap[1]}"))
        cap = min(caps)
    except:
        logger.warning("device_capability is not available. Using 8.0 as default")
        cap = 8.0
    return cap


def configure_attn(
    model_path: str,
    device_id: list[int],
    load_dtype: str | None | torch.dtype,
    trust_remote_code: bool = False,
) -> dict:
    gpu_cap = get_gpu_capability(device_id)
    model_config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    arch_name = getattr(model_config, "architectures", [None])[0]
    cls = getattr(transformers, arch_name, None)

    # do not configure attention for third-party models
    if (cls is None) or trust_remote_code:
        logger.warning(
            f"The attention configuration is not available for model: {arch_name}."
        )
        return {}

    # check code availability
    support_flash = getattr(cls, "_supports_flash_attn_2", False)
    support_sdpa = getattr(cls, "_supports_sdpa", False)

    # check FlashAttention availability
    has_flash_attn = True
    try:
        import flash_attn
    except:
        has_flash_attn = False

    # check dtype compatibility
    if load_dtype not in {torch.float16, torch.bfloat16}:
        if support_flash:
            logger.warning(
                "FlashAttention/Pytorch SDPA only supports float16 and bfloat16. "
                "Please explicitly set `load_dtype` to one of them to enable FlashAttention."
            )
        support_flash = False
        support_sdpa = False

    # set attention implementation
    attn_args = {}
    if support_flash and (gpu_cap >= 8.0) and has_flash_attn:
        attn_args["attn_implementation"] = "flash_attention_2"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.warning("Enable flash_attention_2.")
    elif support_sdpa and (gpu_cap >= 8.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.warning("Enable pytorch flash_attn SDPA kernel.")
    elif support_sdpa and (7.0 <= gpu_cap < 8.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.warning("Enable pytorch memory efficient SDPA kernel.")
        logger.warning("SDPA memory efficient mode does not support bf16.")
    elif support_sdpa and (0 < gpu_cap < 7.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.warning("Enable pytorch math SDPA kernel.")
    else:
        attn_args["attn_implementation"] = "eager"
        logger.info(f"flash attention is not available.")
    return attn_args
