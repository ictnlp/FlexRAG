# Utilities for Huggingface models

from dataclasses import field
from typing import Annotated, Optional

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from flexrag.common import LOGGER_MANAGER, Choices, configure

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
    support_flash = getattr(cls, "_supports_flash_attn", False)
    support_sdpa = getattr(cls, "_supports_sdpa", False)
    support_flex = getattr(cls, "_supports_flex_attn", False)

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
                "FlashAttention/SDPA/FlexAttention only supports float16 and bfloat16. "
                "Please explicitly set `load_dtype` to one of them to enable FlashAttention."
            )
        support_flash = False
        support_sdpa = False
        support_flex = False

    # set attention implementation
    attn_args = {}
    if support_flash and (gpu_cap >= 8.0) and has_flash_attn:
        attn_args["attn_implementation"] = "flash_attention_2"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enable flash_attention_2.")
    elif support_flex and (gpu_cap >= 8.0):
        attn_args["attn_implementation"] = "flex_attention"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enable flex attention.")
    elif support_sdpa and (gpu_cap >= 8.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enable pytorch flash_attn SDPA kernel.")
    elif support_sdpa and (7.0 <= gpu_cap < 8.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enable pytorch memory efficient SDPA kernel.")
        logger.info("SDPA memory efficient mode does not support bf16.")
    elif support_sdpa and (0 < gpu_cap < 7.0):
        attn_args["attn_implementation"] = "sdpa"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        logger.info("Enable pytorch math SDPA kernel.")
    else:
        attn_args["attn_implementation"] = "eager"
        logger.info(f"flash attention is not available.")
    return attn_args


def get_colbert_model(
    base_model: str = "bert",
    output_dim: int = 128,
    model_path: str = None,
):
    """Code adapted from https://github.com/hotchpotch/JQaRA/blob/main/evaluator/reranker/colbert_reranker.py"""
    match base_model:
        case "bert":
            pretrained_class = BertPreTrainedModel
            model_class = BertModel
        case "xlm-roberta":
            pretrained_class = XLMRobertaPreTrainedModel
            model_class = XLMRobertaModel
        case "self_implemented":
            model_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            assert "AutoModel" in model_cfg.auto_map
            model_class_str = model_cfg.auto_map["AutoModel"]
            pretrained_class_str = model_class_str.replace("Model", "PreTrainedModel")
            model_class = get_class_from_dynamic_module(model_class_str, model_path)
            pretrained_class = get_class_from_dynamic_module(
                pretrained_class_str, model_path
            )
        case _:
            raise ValueError(f"Unsupported base model: {base_model}")

    class ColBERTModel(pretrained_class):
        def __init__(self, config):
            super().__init__(config)
            setattr(self, self.base_model_prefix, model_class(config))
            self.linear = torch.nn.Linear(config.hidden_size, output_dim, bias=False)
            self.init_weights()
            return

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
        ):
            outputs = getattr(self, self.base_model_prefix)(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,  # Always output hidden states
            )

            sequence_output = outputs[0]
            return self.linear(sequence_output)

    return ColBERTModel


def load_hf_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    model_type: Optional[str] = None,
    device_id: list[int] = [],
    load_dtype: str = "auto",
    trust_remote_code: bool = False,
    pipeline_parallel: bool = False,
    is_training: bool = False,
    colbert_base_model: str = "bert",
    colbert_dim: int = 128,
    other_model_kwargs: dict = {},
    other_tokenizer_kwargs: dict = {},
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    # prepare dtype
    load_in_4bit = False
    load_in_8bit = False
    match load_dtype:
        case "bfloat16":
            load_dtype = torch.bfloat16
        case "bf16":
            load_dtype = torch.bfloat16
        case "float32":
            load_dtype = torch.float32
        case "fp32":
            load_dtype = torch.float32
        case "float16":
            load_dtype = torch.float16
        case "fp16":
            load_dtype = torch.float16
        case "half":
            load_dtype = torch.float16
        case "8bit":
            load_dtype = None
            load_in_8bit = True
        case "4bit":
            load_dtype = None
            load_in_4bit = True
        case "auto":
            load_dtype = "auto"
        case _:
            raise ValueError(f"Unsupported load_dtype: {load_dtype}")

    # prepare device
    if pipeline_parallel:
        device_map = "auto"
    elif torch.cuda.is_available() and (len(device_id) > 0):
        device_map = device_id[0]
    else:
        device_map = None

    # configure attention implementation
    attn_args = configure_attn(
        model_path=model_path,
        device_id=device_id,
        load_dtype=load_dtype,
        trust_remote_code=trust_remote_code,
    )

    # load model
    match model_type:
        case "causal_lm":
            model_class = AutoModelForCausalLM
        case "seq2seq":
            model_class = AutoModelForSeq2SeqLM
        case "sequence_classification":
            model_class = AutoModelForSequenceClassification
        case "colbert":
            model_class = get_colbert_model(colbert_base_model, colbert_dim, model_path)
        case "masked_lm":
            model_class = AutoModelForMaskedLM
        case "auto":
            model_class = AutoModel
        case "clip":
            model_class = AutoModel
        case "vlm":
            model_class = AutoModelForImageTextToText
        case _:
            model_class = AutoModel
    model = model_class.from_pretrained(
        model_path,
        device_map=device_map,
        dtype=load_dtype,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        trust_remote_code=trust_remote_code,
        **other_model_kwargs,
        **attn_args,
    )

    # patch: some model does not support `int` device_map
    if isinstance(device_map, int):
        model = model.to(torch.device(device_map))

    if not is_training:
        model.eval()

    # load tokenizer
    if tokenizer_path is not None:
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = model_path
    match model_type:
        case "clip":
            tokenizer = (
                AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=trust_remote_code,
                    **other_tokenizer_kwargs,
                ),
                AutoImageProcessor.from_pretrained(
                    tokenizer_path,
                    trust_remote_code=trust_remote_code,
                    **other_tokenizer_kwargs,
                ),
            )
        case "vlm":
            tokenizer = AutoProcessor.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
                **other_tokenizer_kwargs,
            )
        case _:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=trust_remote_code,
                **other_tokenizer_kwargs,
            )
    return model, tokenizer


@configure
class HFModelConfig:
    """The Base Configuration for Huggingface Models,
    including `HFGenerator`, `HFVLMGenerator`, `HFEncoder` and `HFClipEncoder`.

    :param model_path: The path to the model. Required.
    :type model_path: str
    :param tokenizer_path: The path to the tokenizer. None for the same as model_path. Default is None.
    :type tokenizer_path: Optional[str]
    :param trust_remote_code: Whether to trust remote code. Default is False.
    :type trust_remote_code: bool
    :param device_id: The device id to use. [] for using CPU. Default is [].
    :type device_id: list[int]
    :param load_dtype: The dtype to load the model. Default is "auto". Available choices are "bfloat16", "bf16", "float32", "fp32", "float16", "fp16", "half", "8bit", "4bit", "auto",
    :type load_dtype: str
    """

    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    trust_remote_code: bool = False
    device_id: list[int] = field(default_factory=list)
    load_dtype: Annotated[
        str,
        Choices(
            "bfloat16",
            "bf16",
            "float32",
            "fp32",
            "float16",
            "fp16",
            "half",
            "8bit",
            "4bit",
            "auto",
        ),
    ] = "auto"
