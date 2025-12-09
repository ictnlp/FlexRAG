import os
from typing import Optional

import httpx

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, configure
from flexrag.common.base64_utils import file_to_base64, image_to_base64

from .generator_base import GenerationConfig
from .remote_generator_base import RemoteGeneratorBase, RemoteGeneratorBaseConfig


@configure
class AnthropicGeneratorConfig(RemoteGeneratorBaseConfig):
    """Configuration for AnthropicGenerator.

    :param model_name: The name of the model. Required.
    :type model_name: str
    :param base_url: The base url of the API. Defaults to None.
    :type base_url: Optional[str]
    :param api_key: The API key. Defaults to os.environ.get("ANTHROPIC_API_KEY", "EMPTY").
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    :param max_concurrency: The maximum number of concurrent generation requests. Defaults to 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: str = os.environ.get("ANTHROPIC_API_KEY", "EMPTY")
    proxy: Optional[str] = None


class AnthropicGenerator(RemoteGeneratorBase):
    async def _create_client(self, config: AnthropicGeneratorConfig):
        """Create and return the async Anthropic client."""
        from anthropic import AsyncAnthropic

        self._model_name = config.model_name
        if config.proxy is not None:
            httpx_client = httpx.Client(proxies=config.proxy)
        else:
            httpx_client = None
        return AsyncAnthropic(
            api_key=config.api_key,
            base_url=config.base_url,
            http_client=httpx_client,
        )

    @TIME_METER("generator.anthropic_generate")
    async def _async_chat_impl(
        self,
        client,
        message: ChatMessages,
        generation_config: GenerationConfig | None,
    ) -> ChatTurn:
        """Perform the async chat call using the OpenAI client."""
        gen_cfg = self._get_options(generation_config)
        message = [self._turn_to_anthropic(turn) for turn in message]
        response = await client.messages.create(messages=message, **gen_cfg)
        return self._anthropic_to_turn(response)

    async def _async_generate_impl(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig | None,
    ) -> str:
        """Perform the async generate call using the OpenAI client."""
        raise NotImplementedError(
            "AnthropicGenerator does not support generate method."
        )

    def _get_options(self, generation_config: GenerationConfig) -> dict:
        if generation_config is None:
            generation_config = GenerationConfig()
        options = {"model": self._model_name}
        if generation_config.max_new_tokens is not None:
            options["max_tokens"] = generation_config.max_new_tokens
        else:
            options["max_tokens"] = 2**14  # default to 16k tokens
        if generation_config.do_sample:
            options["temperature"] = generation_config.temperature
        else:
            options["temperature"] = 0.0
        # Anthropic models does not support setting both top_k and temperature
        if generation_config.top_k is not None:
            options["top_k"] = generation_config.top_k
        if generation_config.top_p is not None:
            options["top_p"] = generation_config.top_p
        if generation_config.stop_str:
            options["stop_sequences"] = generation_config.stop_str
        return options

    @staticmethod
    def _turn_to_anthropic(turn: ChatTurn) -> dict:
        if isinstance(turn.content, str):
            return {"role": turn.role, "content": turn.content}
        data = {"role": turn.role, "content": []}
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data["content"].append(
                    {
                        "type": "text",
                        "text": content_part.get("text", ""),
                    }
                )
            elif content_part.get("type") == "image":
                if content_part.get("url") is not None:
                    data["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": content_part.get("url"),
                            },
                        }
                    )
                elif content_part.get("image") is not None:
                    base64_image = image_to_base64(
                        content_part.get("image"), format="JPEG"
                    )
                    data["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
                        }
                    )
                elif content_part.get("image_path") is not None:
                    base64_image = file_to_base64(content_part.get("image_path"))
                    data["content"].append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image,
                            },
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
    def _anthropic_to_turn(data) -> ChatTurn:
        return ChatTurn.from_dict(data.to_dict())
