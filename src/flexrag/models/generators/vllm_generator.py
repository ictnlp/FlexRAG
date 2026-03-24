from typing import Annotated, Optional

import PIL

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, Choices, configure
from flexrag.common.logging import LOGGER_MANAGER

from .generator_base import GENERATORS, GenerationConfig, GeneratorBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.vllm")


@configure
class VLLMGeneratorConfig:
    """Configuration for VLLMGenerator.

    :param model_path: Path to the model. Required.
    :type model_path: str
    :param gpu_memory_utilization: Fraction of GPU memory to use. Default to 0.85.
    :type gpu_memory_utilization: float
    :param max_model_len: Maximum length of the model. Defaults to None.
    :type max_model_len: Optional[int]
    :param tensor_parallel: The number of tensor parallel. Defaults to 1.
    :type tensor_parallel: int
    :param load_dtype: The dtype to load the model. Defaults to "auto".
        Available options are "auto", "float32", "float16", "bfloat16".
    :type load_dtype: str
    :enforce_eager: Whether to enforce eager execution. Defaults to False.
    :type enforce_eager: bool
    :trust_remote_code: Whether to trust remote code when loading the model. Defaults to False.
    :type trust_remote_code: bool
    """

    model_path: Optional[str] = None
    gpu_memory_utilization: float = 0.85
    max_model_len: Optional[int] = None
    tensor_parallel: int = 1
    load_dtype: Annotated[
        str,
        Choices("auto", "float32", "float16", "bfloat16"),
    ] = "auto"
    trust_remote_code: bool = False
    enforce_eager: bool = False


@GENERATORS("vllm", config_class=VLLMGeneratorConfig)
class VLLMGenerator(GeneratorBase):
    def __init__(self, cfg: VLLMGeneratorConfig) -> None:
        from vllm import LLM

        assert cfg.model_path is not None, "`model_path` must be provided"

        # load model
        llm_args = {
            "model": cfg.model_path,
            "dtype": str(cfg.load_dtype),
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "tensor_parallel_size": cfg.tensor_parallel,
            "enforce_eager": cfg.enforce_eager,
            "trust_remote_code": cfg.trust_remote_code,
        }
        if cfg.max_model_len is not None:
            llm_args["max_model_len"] = cfg.max_model_len
        self.model = LLM(**llm_args)
        return

    @TIME_METER("generator.vllm_generate")
    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        if not isinstance(prefixes, list):
            prefixes = [prefixes]
        responses = self.model.generate(
            prefixes,
            sampling_params=self._get_options(generation_config),
            use_tqdm=False,
        )
        responses = [[i.text for i in resp.outputs] for resp in responses]
        return responses

    async def async_generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        raise NotImplementedError("VLLMGenerator does not support async_generate yet.")

    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        # Normalize input to list of ChatMessages
        if isinstance(messages, ChatMessages) or isinstance(messages[0], dict):
            messages = [messages]
        for i in range(len(messages)):
            if isinstance(messages[i], list):
                messages[i] = ChatMessages.from_list(messages[i])
        # Process chat requests
        responses = self.model.chat(
            messages=[[self._turn_to_vllm(turn) for turn in msg] for msg in messages],
            sampling_params=self._get_options(generation_config),
            use_tqdm=False,
        )
        return [[self._vllm_to_turn(r) for r in resp.outputs] for resp in responses]

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        raise NotImplementedError("VLLMGenerator does not support async_chat yet.")

    def _get_options(self, generation_config: GenerationConfig | None):
        from vllm import SamplingParams

        if generation_config is None:
            generation_config = GenerationConfig()

        options = {
            "n": generation_config.sample_num,
            "max_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "stop": generation_config.stop_str,
        }
        if generation_config.top_k is None:
            options["top_k"] = 0
        else:
            options["top_k"] = generation_config.top_k
        if generation_config.top_p is not None:
            options["top_p"] = generation_config.top_p
        if generation_config.eos_token_id is not None:
            options["stop_token_ids"] = [generation_config.eos_token_id]
        else:
            options["stop_token_ids"] = [self.model.get_tokenizer().eos_token_id]
        return SamplingParams(**options)

    def _turn_to_vllm(self, turn: ChatTurn):
        if isinstance(turn.content, list):
            for content_part in turn.content:
                assert content_part.get("type") in {
                    "text",
                    "image",
                    "audio",
                    "video",
                }, f"Unsupported content type: {content_part.get('type')}"
        if isinstance(turn.content, str):
            return {"role": turn.role, "content": turn.content}
        data = {"role": turn.role, "content": []}
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data["content"].append(
                    {"type": "text", "text": content_part.get("text", "")}
                )
            elif content_part.get("type") == "image":
                if content_part.get("url") is not None:
                    data["content"].append({"image_url": content_part.get("url")})
                elif content_part.get("image") is not None:
                    img_pil = content_part.get("image")
                    data["content"].append({"image_pil": img_pil})
                elif content_part.get("image_path") is not None:
                    img_pil = PIL.Image.open(content_part.get("image_path"))
                    data["content"].append({"image_pil": img_pil})
                else:
                    raise ValueError(
                        "Image content must have either 'url', 'image', or 'image_path'."
                    )
            elif content_part.get("type") == "audio":
                if content_part.get("url") is not None:
                    data["content"].append({"audio_url": content_part.get("url")})
                else:
                    raise ValueError("Audio content must have 'url'.")
            elif content_part.get("type") == "video":
                if content_part.get("url") is not None:
                    data["content"].append({"video_url": content_part.get("url")})
                else:
                    raise ValueError("Video content must have 'url'.")
            else:
                raise ValueError(
                    f"Unsupported content type: {content_part.get('type')}"
                )
        return data

    def _vllm_to_turn(self, data) -> ChatTurn:
        return ChatTurn(
            role="assistant",
            content=data.text,
            reasoning_content=getattr(data, "reasoning", None),
        )
