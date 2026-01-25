from typing import Annotated

import torch
from transformers import GenerationConfig as HFGenerationConfig
from transformers import PreTrainedModel

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, Choices, configure
from flexrag.common.base64_utils import image_to_base64
from flexrag.common.logging import LOGGER_MANAGER

from ..hf_utils import HFModelConfig, load_hf_model
from .generator_base import GENERATORS, GenerationConfig, GeneratorBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.hf_model")


@configure
class HFGeneratorConfig(HFModelConfig):
    """Configuration for HFGenerator.

    :param pipeline_parallel: Whether to use pipeline parallel. Default is False.
    :type pipeline_parallel: bool
    :param model_type: The type of the model. Default is "causal_lm".
        Available choices are "causal_lm", "seq2seq", "auto", and "vlm".
    :type model_type: str
    """

    pipeline_parallel: bool = False
    model_type: Annotated[
        str,
        Choices("causal_lm", "seq2seq", "vlm", "auto"),
    ] = "causal_lm"


@GENERATORS("hf", config_class=HFGeneratorConfig)
class HFGenerator(GeneratorBase):
    model: PreTrainedModel

    def __init__(self, cfg: HFGeneratorConfig) -> None:
        # load model
        self.model, self.tokenizer = load_hf_model(
            model_path=cfg.model_path,
            tokenizer_path=cfg.tokenizer_path,
            model_type=cfg.model_type,
            device_id=cfg.device_id,
            load_dtype=cfg.load_dtype,
            trust_remote_code=cfg.trust_remote_code,
            pipeline_parallel=cfg.pipeline_parallel,
        )
        self.model_type = cfg.model_type
        self._patch_model()
        return

    @TIME_METER("generator.hf_generate")
    @torch.no_grad()
    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        bsz = len(prefixes)
        inputs = self.tokenizer(
            prefixes, return_tensors="pt", padding=True, truncation=True
        )
        inputs = inputs.to(self.model.device)

        # prepare generation config
        hf_gen_cfg = self._get_options(generation_config)
        sample_num = hf_gen_cfg.num_return_sequences
        if generation_config is not None and generation_config.eos_token_id is not None:
            inputs["eos_token_id"] = generation_config.eos_token_id
        else:
            inputs["eos_token_id"] = self.tokenizer.eos_token_id

        # generate
        if hf_gen_cfg.stop_strings is not None:
            inputs["tokenizer"] = self.tokenizer  # for stop_strings
        outputs = self.model.generate(
            **inputs,
            generation_config=hf_gen_cfg,
        )

        # truncate the input tokens and decode
        if self.model_type == "seq2seq":
            outputs = outputs.view(bsz, sample_num, -1)
            responses = [
                [
                    self.tokenizer.decode(sample, skip_special_tokens=True)
                    for sample in samples
                ]
                for samples in outputs
            ]
        else:
            outputs = outputs.view(bsz, sample_num, -1)
            input_lengths = inputs["attention_mask"].sum(dim=1)
            responses = []
            for i in range(bsz):
                samples = [sample[input_lengths[i] :] for sample in outputs[i]]
                samples = [
                    self.tokenizer.decode(sample, skip_special_tokens=True)
                    for sample in samples
                ]
                responses.append(samples)
        return responses

    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        raise NotImplementedError("HFGenerator does not support async_generate yet.")

    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        # normalize input to list of ChatMessages
        if isinstance(messages, ChatMessages) or isinstance(messages[0], dict):
            messages = [messages]
        for i in range(len(messages)):
            if isinstance(messages[i], list):
                messages[i] = ChatMessages.from_list(messages[i])
        if self.model_type == "vlm":
            messages = [
                [self._turn_to_hf(turn, force_list=True) for turn in msg]
                for msg in messages
            ]
        else:
            messages = [[self._turn_to_hf(turn) for turn in msg] for msg in messages]

        # prepare generation config
        hf_gen_cfg = self._get_options(generation_config)
        bsz = len(messages)
        sample_num = hf_gen_cfg.num_return_sequences

        # prepare inputs
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        # generate responses
        outputs = self.model.generate(**inputs, generation_config=hf_gen_cfg)

        # truncate the input tokens and decode
        outputs = outputs.view(bsz, sample_num, -1)
        input_lengths = inputs["attention_mask"].sum(dim=1)
        responses: list[list[str]] = []
        for i in range(bsz):
            samples = [sample[input_lengths[i] :] for sample in outputs[i]]
            samples = [
                self.tokenizer.decode(sample, skip_special_tokens=True)
                for sample in samples
            ]
            responses.append(samples)
        return [[self._hf_to_turn(r) for r in resp] for resp in responses]

    async def async_chat(
        self,
        prompts: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        raise NotImplementedError("HFGenerator does not support async_chat yet.")

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
        if generation_config.stop_str:  # empty list is not allowed
            cfg.stop_strings = list(generation_config.stop_str)
        return cfg

    def _patch_model(self) -> None:
        if hasattr(self.tokenizer, "tokenizer"):
            tok = self.tokenizer.tokenizer
        else:
            tok = self.tokenizer
        # Add pad token if not exist
        if tok.pad_token_id is None:
            tok.add_special_tokens({"pad_token": "<pad>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
        return

    @staticmethod
    def _turn_to_hf(turn: ChatTurn, force_list: bool = False) -> dict:
        if isinstance(turn.content, str) and not force_list:
            return {"role": turn.role, "content": turn.content}
        elif isinstance(turn.content, str) and force_list:
            return {
                "role": turn.role,
                "content": [{"type": "text", "text": turn.content}],
            }
        data = {"role": turn.role, "content": []}
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data["content"].append(
                    {
                        "type": "text",
                        "text": content_part.get("text", ""),
                    }
                )
            elif content_part.get("type") == "reasoning":
                continue  # skip reasoning parts
            elif content_part.get("type") == "image":
                if content_part.get("url") is not None:
                    data["content"].append(
                        {
                            "type": "image",
                            "url": content_part.get("url"),
                        }
                    )
                elif content_part.get("image") is not None:
                    base64_image = image_to_base64(
                        content_part.get("image"), format="JPEG"
                    )
                    data["content"].append(
                        {
                            "type": "image",
                            "base64": base64_image,
                        }
                    )
                elif content_part.get("image_path") is not None:
                    data["content"].append(
                        {
                            "type": "image",
                            "path": content_part.get("image_path"),
                        }
                    )
                else:
                    raise ValueError(
                        "Image content must have either 'url', 'image', or 'image_path'."
                    )
            else:
                raise ValueError(
                    f"Unsupported content type: {content_part.get('type')}"
                )
        return data

    @staticmethod
    def _hf_to_turn(data: str) -> ChatTurn:
        return ChatTurn(role="assistant", content=data)
