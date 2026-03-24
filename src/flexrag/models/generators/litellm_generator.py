import mimetypes
import os
from dataclasses import field
from typing import Any, Optional
from urllib.parse import urlparse

import litellm

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, configure
from flexrag.common.base64_utils import (
    binary_to_base64,
    file_to_base64,
    image_to_base64,
)
from flexrag.common.logging import LOGGER_MANAGER

from .generator_base import GENERATORS, GenerationConfig
from .remote_generator_base import RemoteGeneratorBase, RemoteGeneratorBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.models.litellm_generator")


def _generation_config_to_kwargs(
    generation_config: GenerationConfig | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if generation_config is None:
        kwargs["temperature"] = 1.0
        return kwargs

    kwargs["temperature"] = (
        generation_config.temperature if generation_config.do_sample else 0.0
    )
    if generation_config.max_new_tokens is not None:
        kwargs["max_tokens"] = generation_config.max_new_tokens
    if generation_config.top_p is not None:
        kwargs["top_p"] = generation_config.top_p
    if generation_config.top_k is not None:
        kwargs["top_k"] = generation_config.top_k
    if generation_config.stop_str:
        kwargs["stop"] = generation_config.stop_str
    return kwargs


def _image_part(content_part: dict[str, Any]) -> dict[str, Any]:
    if content_part.get("url") is not None:
        return {
            "type": "image_url",
            "image_url": {"url": content_part["url"]},
        }
    if content_part.get("image") is not None:
        base64_image = image_to_base64(content_part["image"], format="JPEG")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
    if content_part.get("image_path") is not None:
        image_path = content_part["image_path"]
        mime_type, _ = mimetypes.guess_type(str(image_path))
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type or 'image/jpeg'};base64,{file_to_base64(image_path)}"
            },
        }
    raise ValueError("Image content must have either 'url', 'image', or 'image_path'.")


def _file_part(
    content_part: dict[str, Any],
    *,
    fallback_mime_type: str,
    fallback_file_name: str,
) -> dict[str, Any]:
    file_obj: dict[str, Any] = {}
    if content_part.get("url") is not None:
        file_obj["file_url"] = content_part["url"]
        file_name = os.path.basename(urlparse(content_part["url"]).path)
        file_obj["filename"] = file_name or fallback_file_name
        return {"type": "file", "file": file_obj}
    if content_part.get("file_path") is not None:
        file_path = content_part["file_path"]
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_obj["filename"] = os.path.basename(str(file_path)) or fallback_file_name
        file_obj["file_data"] = (
            f"data:{mime_type or fallback_mime_type};base64,{file_to_base64(file_path)}"
        )
        return {"type": "file", "file": file_obj}
    if content_part.get("binary") is not None:
        file_obj["filename"] = fallback_file_name
        file_obj["file_data"] = (
            f"data:{fallback_mime_type};base64,{binary_to_base64(content_part['binary'])}"
        )
        return {"type": "file", "file": file_obj}
    raise ValueError("File content must have either 'url', 'file_path', or 'binary'.")


def _turn_to_litellm_message(turn: ChatTurn) -> dict[str, Any]:
    if isinstance(turn.content, str):
        return {"role": turn.role, "content": turn.content}

    parts: list[dict[str, Any]] = []
    for content_part in turn.content:
        content_type = content_part.get("type")
        if content_type == "text":
            parts.append({"type": "text", "text": content_part.get("text", "")})
        elif content_type == "reasoning":
            continue
        elif content_type == "image":
            parts.append(_image_part(content_part))
        elif content_type == "pdf":
            parts.append(
                _file_part(
                    content_part,
                    fallback_mime_type="application/pdf",
                    fallback_file_name="document.pdf",
                )
            )
        elif content_type == "audio":
            parts.append(
                _file_part(
                    content_part,
                    fallback_mime_type="audio/mpeg",
                    fallback_file_name="audio.mp3",
                )
            )
        elif content_type == "video":
            parts.append(
                _file_part(
                    content_part,
                    fallback_mime_type="video/mp4",
                    fallback_file_name="video.mp4",
                )
            )
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
    return {"role": turn.role, "content": parts}


def _choice_message_content(response: Any) -> Any:
    if isinstance(response, dict):
        return response["choices"][0]["message"].get("content")

    choices = getattr(response, "choices", None)
    if choices is None:
        raise ValueError("LiteLLM completion response does not contain choices.")
    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message")
    if message is None:
        raise ValueError("LiteLLM completion response does not contain a message.")
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _completion_response_to_chat_turn(response: Any) -> ChatTurn:
    content = _choice_message_content(response)
    if isinstance(content, str):
        return ChatTurn(role="assistant", content=content)

    if not isinstance(content, list):
        return ChatTurn(role="assistant", content="")

    normalized_parts: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "output_text"}:
            normalized_parts.append({"type": "text", "text": part.get("text", "")})
        elif part_type in {"reasoning", "reasoning_text"}:
            normalized_parts.append({"type": "reasoning", "text": part.get("text", "")})
        elif part_type == "image_url":
            image_url = part.get("image_url", {})
            if isinstance(image_url, dict):
                normalized_parts.append(
                    {"type": "image", "url": image_url.get("url", "")}
                )
        elif part_type == "file":
            file_part = part.get("file", {})
            if isinstance(file_part, dict):
                file_url = file_part.get("file_url")
                if file_url:
                    normalized_parts.append({"type": "pdf", "url": file_url})

    if not normalized_parts:
        return ChatTurn(role="assistant", content="")

    if all(part["type"] == "text" for part in normalized_parts):
        return ChatTurn(
            role="assistant",
            content="".join(part.get("text", "") for part in normalized_parts),
        )
    return ChatTurn(role="assistant", content=normalized_parts)


def _text_completion_response_to_text(response: Any) -> str:
    if isinstance(response, dict):
        choice = response["choices"][0]
        if "text" in choice:
            return choice["text"]
        return choice["message"]["content"]

    choices = getattr(response, "choices", None)
    if choices is None:
        raise ValueError("LiteLLM text completion response does not contain choices.")
    choice = choices[0]
    if hasattr(choice, "text") and choice.text is not None:
        return choice.text
    if isinstance(choice, dict) and choice.get("text") is not None:
        return choice["text"]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")
    if message is None:
        raise ValueError("LiteLLM text completion response does not contain text.")
    if isinstance(message, dict):
        return message.get("content", "")
    return getattr(message, "content", "")


@configure
class LiteLLMGeneratorConfig(RemoteGeneratorBaseConfig):
    """Configuration for LiteLLMGenerator.

    :param provider: LiteLLM provider prefix, e.g. ``openai`` or ``ollama``.
    :type provider: Optional[str]
    :param model_name: Provider-specific model identifier without the provider prefix.
    :type model_name: Optional[str]
    :param api_key: API key passed to LiteLLM as ``api_key``. Defaults to None.
    :type api_key: Optional[str]
    :param base_url: Base URL passed to LiteLLM as ``api_base``. Defaults to None.
    :type base_url: Optional[str]
    :param api_version: Provider API version passed through to LiteLLM. Defaults to None.
    :type api_version: Optional[str]
    :param timeout: Request timeout in seconds. Defaults to None.
    :type timeout: Optional[float]
    :param proxy: Upstream proxy setting forwarded to LiteLLM. Defaults to None.
    :type proxy: Optional[str]
    :param extra_kwargs: Additional provider-specific LiteLLM request kwargs.
        Explicit top-level config fields take precedence over conflicting keys here.
    :type extra_kwargs: dict[str, Any]

    For Example, Calling LLMs from OpenRouter with LiteLLMGenerator:

        >>> config = LiteLLMGeneratorConfig(
        ...     provider="openrouter",
        ...     model_name="gpt-5.4",
        ...     api_key=os.getenv("OPEN_ROUTER_API_KEY"),
        ... )
        >>> generator = LiteLLMGenerator(config)

    Or calling LLMs from OpenAI with LiteLLMGenerator:

        >>> config = LiteLLMGeneratorConfig(
        ...     provider="openai",
        ...     model_name="gpt-5.4",
        ...     api_key=os.getenv("OPENAI_API_KEY"),
        ... )
        >>> generator = LiteLLMGenerator(config)

    Or calling LLMs from a custom OpenAI-compatible endpoint with LiteLLMGenerator:

        >>> config = LiteLLMGeneratorConfig(
        ...     provider="openai",
        ...     model_name="your-model-name",
        ...     base_url="https://your-custom-endpoint.com",
        ...     api_key=os.getenv("YOUR_CUSTOM_ENDPOINT_API_KEY"),
        ... )
        >>> generator = LiteLLMGenerator(config)
    """

    provider: Optional[str] = None
    model_name: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[float] = None
    proxy: Optional[str] = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@GENERATORS("litellm", config_class=LiteLLMGeneratorConfig)
class LiteLLMGenerator(RemoteGeneratorBase):
    async def _create_client(self, config: LiteLLMGeneratorConfig):
        provider = (config.provider or "").strip()
        model_name = (config.model_name or "").strip()
        assert provider, "`provider` must be provided for LiteLLM models."
        assert model_name, "`model_name` must be provided for LiteLLM models."

        request_kwargs = dict(config.extra_kwargs)
        explicit_kwargs = {
            "api_key": config.api_key,
            "api_base": config.base_url,
            "api_version": config.api_version,
            "timeout": config.timeout,
            "proxy": config.proxy,
        }
        request_kwargs.update(
            {key: value for key, value in explicit_kwargs.items() if value is not None}
        )
        return {
            "model": f"{provider}/{model_name}",
            "request_kwargs": request_kwargs,
        }

    @TIME_METER("generator.litellm_chat")
    async def _async_chat_impl(
        self,
        client,
        message: ChatMessages,
        generation_config: GenerationConfig | None,
    ) -> ChatTurn:
        request_kwargs = dict(client["request_kwargs"])
        request_kwargs.update(_generation_config_to_kwargs(generation_config))
        request_kwargs["model"] = client["model"]
        request_kwargs["messages"] = [
            _turn_to_litellm_message(turn) for turn in message
        ]
        response = await litellm.acompletion(**request_kwargs)
        return _completion_response_to_chat_turn(response)

    @TIME_METER("generator.litellm_generate")
    async def _async_generate_impl(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig | None,
    ) -> str:
        if not hasattr(litellm, "atext_completion"):
            raise NotImplementedError(
                "The installed LiteLLM version does not provide `atext_completion`."
            )
        request_kwargs = dict(client["request_kwargs"])
        request_kwargs.update(_generation_config_to_kwargs(generation_config))
        request_kwargs["model"] = client["model"]
        request_kwargs["prompt"] = prompt
        response = await litellm.atext_completion(**request_kwargs)
        return _text_completion_response_to_text(response)
