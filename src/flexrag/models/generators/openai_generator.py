import os
from typing import Optional

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI, omit
from openai.types.responses import Response

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, configure
from flexrag.common.base64_utils import (
    binary_to_base64,
    file_to_base64,
    image_to_base64,
)
from flexrag.common.logging import LOGGER_MANAGER

from .generator_base import GENERATORS, GenerationConfig
from .remote_generator_base import RemoteGeneratorBase, RemoteGeneratorBaseConfig

logger = LOGGER_MANAGER.get_logger("flexrag.models.openai_generator")


@configure
class OpenAIGeneratorConfig(RemoteGeneratorBaseConfig):
    """Configuration for OpenAI Generator.

    :param is_azure: Whether the model is hosted on Azure. Default is False.
    :type is_azure: bool
    :param model_name: The name of the model to use.
    :type model_name: str
    :param base_url: The base URL of the OpenAI API. Default is None.
    :type base_url: Optional[str]
    :param api_key: The API key for OpenAI. Default is os.environ.get("OPENAI_API_KEY", "EMPTY").
    :type api_key: str
    :param api_version: The API version to use. Default is "2024-07-01-preview".
    :type api_version: str
    :param timeout: The timeout for the HTTP client in seconds. Default is None.
    :type timeout: Optional[float]
    :param proxy: The proxy to use for the HTTP client. Default is None.
    :type proxy: Optional[str]
    :param max_concurrency: The maximum number of concurrent generation requests. Default is 1.
    :type max_concurrency: int
    """

    is_azure: bool = False
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    api_key: str = os.environ.get("OPENAI_API_KEY", "EMPTY")
    api_version: str = "2024-07-01-preview"
    timeout: Optional[float] = None
    proxy: Optional[str] = None


@GENERATORS("openai", config_class=OpenAIGeneratorConfig)
class OpenAIGenerator(RemoteGeneratorBase):

    async def _create_client(self, config: OpenAIGeneratorConfig):
        """Create and return the async OpenAI client."""
        # Run inside background event loop
        self._model_name = config.model_name
        if config.proxy is not None:
            httpx_client = httpx.AsyncClient(proxies=config.proxy)
        else:
            httpx_client = None
        if config.is_azure:
            return AsyncAzureOpenAI(
                api_key=config.api_key,
                api_version=config.api_version,
                azure_endpoint=config.base_url,
                http_client=httpx_client,
                timeout=config.timeout,
            )
        else:
            return AsyncOpenAI(
                base_url=config.base_url,
                api_key=config.api_key,
                http_client=httpx_client,
                timeout=config.timeout,
            )

    @TIME_METER("generator.openai_generate")
    async def _async_chat_impl(
        self,
        client: AsyncOpenAI | AsyncAzureOpenAI,
        message: ChatMessages,
        generation_config: GenerationConfig,
    ) -> ChatTurn:
        """Perform the async chat call using the OpenAI client."""
        gen_cfg = self._get_options(generation_config)
        message = [self._turn_to_openai(turn) for turn in message]
        response = await client.responses.create(input=message, **gen_cfg)
        return self._openai_to_turn(response)

    @TIME_METER("generator.openai_generate")
    async def _async_generate_impl(
        self,
        client: AsyncOpenAI | AsyncAzureOpenAI,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        """Perform the async generate call using the OpenAI client."""
        gen_cfg = self._get_options(generation_config)
        if "max_output_tokens" in gen_cfg:
            gen_cfg["max_tokens"] = gen_cfg.pop("max_output_tokens")
        response = await client.completions.create(prompt=prompt, **gen_cfg)
        return response.choices[0].text

    def _get_options(self, generation_config: GenerationConfig | None) -> dict:
        """Get generation options from GenerationConfig."""
        if generation_config is None:
            generation_config = GenerationConfig()
        # hotfix for vllm deployed llama-3
        if "llama-3" in self._model_name.lower():
            extra_body = {"stop_token_ids": [128009]}
        else:
            extra_body = None
        if generation_config.top_p is None:
            top_p = omit
        else:
            top_p = generation_config.top_p
        if generation_config.max_new_tokens is None:
            max_new_tokens = omit
        else:
            max_new_tokens = generation_config.max_new_tokens
        options = {
            "extra_body": extra_body,
            "top_p": top_p,
            "max_output_tokens": max_new_tokens,
            "model": self._model_name,
        }
        if generation_config.do_sample:
            options["temperature"] = generation_config.temperature
        else:
            options["temperature"] = 0.0
        if generation_config.stop_str:
            logger.warning("`stop_str` is not supported in OpenAI response API.")
        return options

    @staticmethod
    def _turn_to_openai(turn: ChatTurn) -> dict[str, str | list[dict]]:
        """Convert ChatTurn to OpenAI message format."""
        if isinstance(turn.content, str):
            return {"role": turn.role, "content": turn.content}
        data = {"role": turn.role, "content": []}
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data["content"].append(
                    {
                        "type": "input_text",
                        "text": content_part.get("text", ""),
                    }
                )
            elif content_part.get("type") == "reasoning":
                continue  # skip reasoning parts
            elif content_part.get("type") == "image":
                if content_part.get("url") is not None:
                    data["content"].append(
                        {
                            "type": "input_image",
                            "image_url": content_part.get("url"),
                        }
                    )
                elif content_part.get("image") is not None:
                    base64_image = image_to_base64(
                        content_part.get("image"), format="JPEG"
                    )
                    data["content"].append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
                elif content_part.get("image_path") is not None:
                    base64_image = file_to_base64(content_part.get("image_path"))
                    data["content"].append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    )
                else:
                    raise ValueError(
                        "Image content must have either 'url', 'image', or 'image_path'."
                    )
            elif content_part.get("type") == "pdf":
                if content_part.get("url") is not None:
                    data["content"].append(
                        {
                            "type": "input_file",
                            "file_url": content_part.get("url"),
                        }
                    )
                elif content_part.get("file_path") is not None:
                    base64_file = file_to_base64(content_part.get("file_path"))
                    data["content"].append(
                        {
                            "type": "input_file",
                            "file_url": f"data:application/pdf;base64,{base64_file}",
                        }
                    )
                elif content_part.get("binary") is not None:
                    base64_file = binary_to_base64(content_part.get("binary"))
                    data["content"].append(
                        {
                            "type": "input_file",
                            "file_url": f"data:application/pdf;base64,{base64_file}",
                        }
                    )
                else:
                    raise ValueError(
                        "File content must have either 'url', 'file_path', or 'binary'."
                    )
            else:
                raise ValueError(
                    f"Unsupported content type: {content_part.get('type')}"
                )
        return data

    @staticmethod
    def _openai_to_turn(data: Response) -> ChatTurn:
        """Convert OpenAI response to ChatTurn."""
        contents = []
        for outputs in data.output:
            data_dict = outputs.to_dict()
            for item in data_dict["content"]:
                if item["type"] == "output_text":
                    item["type"] = "text"
                if item["type"] == "reasoning_text":
                    item["type"] = "reasoning"
                if item["type"] not in {"text", "image", "pdf", "reasoning"}:
                    raise ValueError(f"Unsupported content type: {item['type']}")
                contents.append(item)
        return ChatTurn.from_dict({"role": "assistant", "content": contents})
