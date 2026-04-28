import json
import mimetypes
import os
from dataclasses import field
from typing import Any, Optional
from urllib.parse import urlparse

import litellm

from flexrag.common import ChatMessages, ChatTurn, ContentPart, configure, trace
from flexrag.common.base64_utils import (
    binary_to_base64,
    file_to_base64,
    image_to_base64,
)
from flexrag.common.logging import LOGGER_MANAGER

from .generator_base import GENERATORS, GenerationConfig, GeneratorBase

logger = LOGGER_MANAGER.get_logger("flexrag.models.litellm_generator")

litellm.suppress_debug_info = True


def _generation_config_to_kwargs(
    generation_config: GenerationConfig | None,
    *,
    chat: bool = False,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if generation_config is None:
        kwargs["temperature"] = 1.0
        return kwargs

    kwargs["n"] = generation_config.sample_num
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
    if chat and generation_config.tools:
        kwargs["tools"] = generation_config.tools
    if chat and generation_config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = generation_config.reasoning_effort
    if generation_config.response_format is not None:
        if chat:
            kwargs["response_format"] = generation_config.response_format
        else:
            logger.warning(
                "LiteLLMGenerator.generate does not support response_format. "
                "This field will be ignored for generate calls."
            )
    return kwargs


def _image_part(content_part: ContentPart) -> dict[str, Any]:
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
    content_part: ContentPart,
    *,
    fallback_mime_type: str,
    fallback_file_name: str,
) -> dict[str, Any]:
    file_obj: dict[str, Any] = {}
    explicit_mime_type = content_part.get("mime_type")
    explicit_file_name = content_part.get("file_name")
    if content_part.get("url") is not None:
        file_obj["file_url"] = content_part["url"]
        file_name = os.path.basename(urlparse(content_part["url"]).path)
        file_obj["filename"] = explicit_file_name or file_name or fallback_file_name
        return {"type": "file", "file": file_obj}
    if content_part.get("file_path") is not None:
        file_path = content_part["file_path"]
        mime_type, _ = mimetypes.guess_type(str(file_path))
        file_obj["filename"] = (
            explicit_file_name or os.path.basename(str(file_path)) or fallback_file_name
        )
        file_obj["file_data"] = (
            f"data:{explicit_mime_type or mime_type or fallback_mime_type};base64,"
            f"{file_to_base64(file_path)}"
        )
        return {"type": "file", "file": file_obj}
    if content_part.get("binary") is not None:
        file_obj["filename"] = explicit_file_name or fallback_file_name
        file_obj["file_data"] = (
            "data:"
            f"{explicit_mime_type or fallback_mime_type};base64,"
            f"{binary_to_base64(content_part['binary'])}"
        )
        return {"type": "file", "file": file_obj}
    raise ValueError("File content must have either 'url', 'file_path', or 'binary'.")


def _parse_tool_arguments(arguments: Any) -> Any:
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return arguments


def _tool_call_block(tool_call: Any) -> dict[str, Any]:
    function_data = tool_call.function
    return {
        "type": "tool_call",
        "id": tool_call.id,
        "name": function_data.name,
        "arguments": _parse_tool_arguments(function_data.arguments),
    }


def _tool_call_payload(tool_call: ContentPart) -> dict[str, Any]:
    arguments = tool_call.get("arguments")
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": tool_call.get("id"),
        "type": "function",
        "function": {
            "name": tool_call.get("name", ""),
            "arguments": arguments if arguments is not None else "",
        },
    }


def _turn_to_litellm_message(turn: ChatTurn) -> dict[str, Any]:
    if turn.role == "tool":
        message: dict[str, Any] = {"role": "tool", "content": turn.content}
        if turn.tool_call_id is not None:
            message["tool_call_id"] = turn.tool_call_id
        if turn.name is not None:
            message["name"] = turn.name
        return message
    if isinstance(turn.content, str):
        return {"role": turn.role, "content": turn.content}

    parts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for content_part in turn.content:
        content_type = content_part.get("type")
        if content_type == "text":
            parts.append({"type": "text", "text": content_part.get("text", "")})
        elif content_type == "tool_call":
            tool_calls.append(_tool_call_payload(content_part))
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
        elif content_type == "file":
            parts.append(
                _file_part(
                    content_part,
                    fallback_mime_type="application/octet-stream",
                    fallback_file_name="document.bin",
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
    message: dict[str, Any] = {
        "role": turn.role,
        "content": parts if parts else (None if tool_calls else ""),
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _completion_choice_to_chat_turn(choice: Any, usage: Any) -> ChatTurn:
    message = choice.message
    content = message.content
    normalized_parts: list[ContentPart] = []
    metadata: dict[str, Any] = {}
    reasoning_content = getattr(message, "reasoning_content", None)
    thinking_blocks = getattr(message, "thinking_blocks", None)

    finish_reason = choice.finish_reason
    if finish_reason is not None:
        metadata["finish_reason"] = finish_reason

    if usage is not None:
        usage_dict = {
            key: value
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if (value := getattr(usage, key, None)) is not None
        }
        if usage_dict:
            metadata["usage"] = usage_dict

    if isinstance(content, str):
        if content:
            normalized_parts.append({"type": "text", "text": content})
    elif content is not None:
        for part in content:
            part_type = part["type"]
            if part_type in {"text", "output_text"}:
                normalized_parts.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                normalized_parts.append(
                    {"type": "image", "url": part["image_url"]["url"]}
                )
            elif part_type == "file":
                file_url = part["file"].get("file_url")
                if file_url:
                    normalized_parts.append(
                        {
                            "type": "file",
                            "url": file_url,
                            "file_name": part["file"].get("filename", ""),
                        }
                    )
            else:
                raise ValueError(f"Unsupported LiteLLM content type: {part_type}")

    message_tool_calls = message.tool_calls or []
    if message_tool_calls:
        normalized_parts.extend(_tool_call_block(call) for call in message_tool_calls)

    if not normalized_parts:
        return ChatTurn(
            role="assistant",
            content="",
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
            metadata=metadata,
        )

    if all(part.get("type") == "text" for part in normalized_parts):
        return ChatTurn(
            role="assistant",
            content="".join(part.get("text", "") for part in normalized_parts),
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
            metadata=metadata,
        )
    return ChatTurn(
        role="assistant",
        content=normalized_parts,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
        metadata=metadata,
    )


def _completion_response_to_chat_turns(response: Any) -> list[ChatTurn]:
    usage = getattr(response, "usage", None)
    return [
        _completion_choice_to_chat_turn(choice, usage) for choice in response.choices
    ]


def _text_completion_choice_to_text(choice: Any) -> str:
    if getattr(choice, "text", None) is not None:
        return choice.text
    return choice.message.content


def _text_completion_response_to_texts(response: Any) -> list[str]:
    return [_text_completion_choice_to_text(choice) for choice in response.choices]


@configure
class LiteLLMGeneratorConfig:
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
    max_concurrency: int = 1
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    timeout: Optional[float] = None
    proxy: Optional[str] = None
    extra_kwargs: dict[str, Any] = field(default_factory=dict)


@GENERATORS("litellm", config_class=LiteLLMGeneratorConfig)
class LiteLLMGenerator(GeneratorBase[LiteLLMGeneratorConfig]):
    def __init__(self, config: LiteLLMGeneratorConfig):
        super().__init__(config)
        return

    def _get_max_concurrency(self) -> int:
        return max(1, self._config.max_concurrency)

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

    @trace("generator.litellm_chat")
    async def _async_chat_one(
        self,
        client,
        message: ChatMessages,
        generation_config: GenerationConfig | None,
    ) -> list[ChatTurn]:
        request_kwargs = dict(client["request_kwargs"])
        request_kwargs.update(
            _generation_config_to_kwargs(generation_config, chat=True)
        )
        request_kwargs["model"] = client["model"]
        request_kwargs["messages"] = [
            _turn_to_litellm_message(turn) for turn in message
        ]
        response = await litellm.acompletion(**request_kwargs)
        return _completion_response_to_chat_turns(response)

    @trace("generator.litellm_generate")
    async def _async_generate_one(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig | None,
    ) -> list[str]:
        request_kwargs = dict(client["request_kwargs"])
        request_kwargs.update(_generation_config_to_kwargs(generation_config))
        request_kwargs["model"] = client["model"]
        request_kwargs["prompt"] = prompt
        response = await litellm.atext_completion(**request_kwargs)
        return _text_completion_response_to_texts(response)

    async def _async_chat_impl(
        self,
        client,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None,
    ) -> list[list[ChatTurn]]:
        return [
            await self._async_chat_one(client, message, generation_config)
            for message in messages
        ]

    async def _async_generate_impl(
        self,
        client,
        prefixes: list[str],
        generation_config: GenerationConfig | None,
    ) -> list[list[str]]:
        return [
            await self._async_generate_one(client, prompt, generation_config)
            for prompt in prefixes
        ]
