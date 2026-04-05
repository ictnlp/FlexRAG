from typing import Annotated

import torch
from transformers import AutoConfig
from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedModel

from flexrag.common import ChatMessages, ChatTurn, Choices, configure, trace
from flexrag.common.logging import LOGGER_MANAGER

from ..hf_utils import HFModelConfig, load_hf_model
from .generator_base import GENERATORS, GenerationConfig
from .local_process_generator_base import LocalProcessGeneratorBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.hf_model")


def _config_supports_multimodal(model_config) -> bool:
    multimodal_attrs = (
        "vision_config",
        "audio_config",
        "video_config",
        "image_config",
    )
    if any(getattr(model_config, attr, None) is not None for attr in multimodal_attrs):
        return True

    sub_configs = getattr(model_config, "sub_configs", None)
    if isinstance(sub_configs, dict) and any(
        any(token in key.lower() for token in ("vision", "image", "audio", "video"))
        for key in sub_configs
    ):
        return True

    model_type = getattr(model_config, "model_type", "") or ""
    if any(
        token in model_type.lower()
        for token in ("vision", "image", "audio", "video", "_vl", "llava", "omni")
    ):
        return True

    architectures = getattr(model_config, "architectures", []) or []
    return any(
        any(
            token in arch.lower()
            for token in ("vision", "image", "audio", "video", "llava", "omni", "vl")
        )
        for arch in architectures
    )


def _resolve_model_type(cfg: "HFGeneratorConfig") -> str:
    if cfg.model_type != "auto":
        return cfg.model_type

    model_config = AutoConfig.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if _config_supports_multimodal(model_config):
        return "vlm"
    if getattr(model_config, "is_encoder_decoder", False):
        return "seq2seq"
    return "causal_lm"


def _content_part_to_hf(content_part: dict) -> dict:
    content_type = content_part.get("type")
    if content_type == "text":
        return {"type": "text", "text": content_part.get("text", "")}
    if content_type == "image":
        if content_part.get("url") is not None:
            return {"type": "image", "url": content_part.get("url")}
        if content_part.get("image") is not None:
            return {"type": "image", "image": content_part.get("image")}
        if content_part.get("image_path") is not None:
            return {"type": "image", "path": content_part.get("image_path")}
        raise ValueError(
            "Image content must have either 'url', 'image', or 'image_path'."
        )
    if content_type == "audio":
        if content_part.get("url") is not None:
            return {"type": "audio", "url": content_part.get("url")}
        if content_part.get("file_path") is not None:
            return {"type": "audio", "path": content_part.get("file_path")}
        if content_part.get("binary") is not None:
            return {"type": "audio", "audio": content_part.get("binary")}
        raise ValueError(
            "Audio content must have either 'url', 'file_path', or 'binary'."
        )
    if content_type == "video":
        if content_part.get("url") is not None:
            return {"type": "video", "url": content_part.get("url")}
        if content_part.get("file_path") is not None:
            return {"type": "video", "path": content_part.get("file_path")}
        if content_part.get("binary") is not None:
            return {"type": "video", "video": content_part.get("binary")}
        raise ValueError(
            "Video content must have either 'url', 'file_path', or 'binary'."
        )
    if content_type == "pdf":
        raise ValueError("HFGenerator does not support pdf content in chat messages.")
    if content_type == "tool_call":
        raise ValueError("HFGenerator does not support native tool_call message parts.")
    raise ValueError(f"Unsupported content type: {content_type}")


def _messages_have_multimodal_content(messages: list[ChatMessages]) -> bool:
    for message in messages:
        for turn in message:
            if isinstance(turn.content, list) and any(
                part.get("type") != "text" for part in turn.content
            ):
                return True
    return False


def _turn_to_hf(turn: ChatTurn, *, force_rich_content: bool = False) -> dict:
    if isinstance(turn.content, str):
        if not force_rich_content:
            return {"role": turn.role, "content": turn.content}
        return {
            "role": turn.role,
            "content": [{"type": "text", "text": turn.content}],
        }

    return {
        "role": turn.role,
        "content": [_content_part_to_hf(content_part) for content_part in turn.content],
    }


@configure
class HFGeneratorConfig(HFModelConfig):
    """Configuration for HFGenerator."""

    pipeline_parallel: bool = False
    model_type: Annotated[
        str,
        Choices("causal_lm", "seq2seq", "auto"),
    ] = "auto"


class HFGeneratorImpl:
    model: PreTrainedModel

    def __init__(self, cfg: HFGeneratorConfig) -> None:
        self._resolved_model_type = _resolve_model_type(cfg)
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type=self._resolved_model_type,
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
            pipeline_parallel=cfg.pipeline_parallel,
        )
        self._supports_multimodal = self._resolved_model_type == "vlm"
        self._patch_model()
        return

    @property
    def _text_tokenizer(self):
        if hasattr(self.tokenizer, "tokenizer"):
            return self.tokenizer.tokenizer
        return self.tokenizer

    def _decode_sample(self, sample) -> str:
        return self._text_tokenizer.decode(sample, skip_special_tokens=True)

    def _get_eos_token_id(self):
        return self._text_tokenizer.eos_token_id

    def _prepare_text_inputs(self, prefixes: list[str]):
        inputs = self.tokenizer(
            prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return inputs.to(self.model.device)

    def _decode_outputs(self, outputs, inputs, sample_num: int) -> list[list[str]]:
        bsz = len(inputs["input_ids"])
        outputs = outputs.view(bsz, sample_num, -1)
        if self._resolved_model_type == "seq2seq":
            return [
                [self._decode_sample(sample) for sample in samples]
                for samples in outputs
            ]

        input_lengths = inputs["attention_mask"].sum(dim=1)
        responses = []
        for i in range(bsz):
            samples = [sample[input_lengths[i] :] for sample in outputs[i]]
            responses.append([self._decode_sample(sample) for sample in samples])
        return responses

    def _get_options(
        self, generation_config: GenerationConfig | None
    ) -> HFGenerationConfig:
        if generation_config is None:
            generation_config = GenerationConfig()
        cfg = HFGenerationConfig(
            do_sample=generation_config.do_sample,
            temperature=generation_config.temperature,
            max_new_tokens=generation_config.max_new_tokens,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            num_return_sequences=generation_config.sample_num,
        )
        if generation_config.stop_str:
            cfg.stop_strings = list(generation_config.stop_str)
        return cfg

    def _patch_model(self) -> None:
        tok = self._text_tokenizer
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(tok))
        return

    @trace("generator.hf_generate")
    @torch.no_grad()
    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        prefixes = prefixes if isinstance(prefixes, list) else [prefixes]
        inputs = self._prepare_text_inputs(prefixes)

        hf_gen_cfg = self._get_options(generation_config)
        sample_num = hf_gen_cfg.num_return_sequences
        inputs["eos_token_id"] = (
            generation_config.eos_token_id
            if generation_config is not None
            and generation_config.eos_token_id is not None
            else self._get_eos_token_id()
        )
        if hf_gen_cfg.stop_strings is not None:
            inputs["tokenizer"] = self._text_tokenizer

        outputs = self.model.generate(**inputs, generation_config=hf_gen_cfg)
        return self._decode_outputs(outputs, inputs, sample_num)

    @trace("generator.hf_chat")
    @torch.no_grad()
    def chat(
        self,
        messages: list[ChatMessages] | ChatMessages,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        if isinstance(messages, ChatMessages):
            normalized_messages = [messages]
        else:
            if not all(isinstance(message, ChatMessages) for message in messages):
                raise TypeError(
                    "HFGeneratorImpl.chat expects normalized ChatMessages batches."
                )
            normalized_messages = messages

        use_multimodal = _messages_have_multimodal_content(normalized_messages)
        if use_multimodal and not self._supports_multimodal:
            raise ValueError(
                "Current HFGenerator instance does not support multimodal chat inputs. "
                'Use `model_type="auto"` with a multimodal-capable model.'
            )

        hf_messages = [
            [_turn_to_hf(turn, force_rich_content=use_multimodal) for turn in msg]
            for msg in normalized_messages
        ]

        hf_gen_cfg = self._get_options(generation_config)
        sample_num = hf_gen_cfg.num_return_sequences
        inputs = self.tokenizer.apply_chat_template(
            hf_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        if generation_config is not None and generation_config.eos_token_id is not None:
            inputs["eos_token_id"] = generation_config.eos_token_id
        elif "eos_token_id" not in inputs:
            inputs["eos_token_id"] = self._get_eos_token_id()
        if hf_gen_cfg.stop_strings is not None:
            inputs["tokenizer"] = self._text_tokenizer

        outputs = self.model.generate(**inputs, generation_config=hf_gen_cfg)
        responses = self._decode_outputs(outputs, inputs, sample_num)
        return [
            [ChatTurn(role="assistant", content=text) for text in resp]
            for resp in responses
        ]


@GENERATORS("hf", config_class=HFGeneratorConfig)
class HFGenerator(LocalProcessGeneratorBase):
    impl_cls = HFGeneratorImpl
