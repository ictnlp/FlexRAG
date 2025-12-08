import io
import os
from typing import Optional

import httpx

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, configure

from .api_based_generator import APIBasedGeneratorBase, APIBasedGeneratorBaseConfig
from .generator_base import GenerationConfig


@configure
class GoogleGeneratorConfig(APIBasedGeneratorBaseConfig):
    """Configuration for GoogleGenerator.

    :param model_name: The name of the model. Required.
    :type model_name: str
    :param vertexai: Whether to use Vertex AI. Defaults to False.
    :type vertexai: bool
    :param base_url: The base url of the API. Defaults to None.
    :type base_url: Optional[str]
    :param api_key: The API key. Defaults to os.environ.get("GOOGLE_API_KEY", "EMPTY").
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    :param max_concurrency: The maximum number of concurrent generation requests. Defaults to 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    vertexai: bool = False
    base_url: Optional[str] = None
    api_key: str = os.environ.get("GOOGLE_API_KEY", "EMPTY")
    proxy: Optional[str] = None


class GoogleGenerator(APIBasedGeneratorBase):
    async def _create_client(self, config: GoogleGeneratorConfig):
        from google import genai

        # initialize the client
        self._model_name = config.model_name
        if config.proxy is not None:
            client = httpx.Client(proxies=config.proxy)
        else:
            client = None
        return genai.Client(
            api_key=config.api_key,
            vertexai=config.vertexai,
            http_options={
                "base_url": config.base_url,
                "httpx_client": client,
            },
        ).aio

    @TIME_METER("generator.google_generate")
    async def _async_chat_impl(
        self,
        client,
        message: ChatMessages,
        generation_config: GenerationConfig | None,
    ) -> ChatTurn:
        message = [self._turn_to_google(turn) for turn in message]
        options = self._get_options(generation_config)
        response = await client.models.generate_content(contents=message, **options)
        return self._google_to_turn(response)

    async def _async_generate_impl(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        raise NotImplementedError(
            "GoogleGenerator does not support text-only generation."
        )

    def _get_options(self, generation_config: GenerationConfig | None) -> dict:
        if generation_config is None:
            generation_config = GenerationConfig()
        options = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "max_output_tokens": generation_config.max_new_tokens,
            "stop_sequences": generation_config.stop_str,
        }
        if generation_config.do_sample:
            options["temperature"] = generation_config.temperature
        else:
            options["temperature"] = 0.0
        return {"config": options, "model": self._model_name}

    @staticmethod
    def _turn_to_google(turn: ChatTurn):
        from google.genai import types

        if isinstance(turn.content, str):
            return types.Content(
                role="user", parts=[types.Part.from_text(text=turn.content)]
            )

        data: types.Content = types.Content(role="user", parts=[])
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data.parts.append(
                    types.Part.from_text(text=content_part.get("text", ""))
                )
            elif content_part.get("type") == "image":
                if content_part.get("url") is not None:
                    data.parts.append(
                        types.Part.from_uri(file_uri=content_part.get("url"))
                    )
                elif content_part.get("image") is not None:
                    buffer = io.BytesIO()
                    content_part.get("image").save(buffer, format="JPEG")
                    data.parts.append(
                        types.Part.from_bytes(
                            data=buffer.getvalue(), mime_type="image/jpeg"
                        )
                    )
                elif content_part.get("image_path") is not None:
                    with open(content_part.get("image_path"), "rb") as f:
                        data.parts.append(
                            types.Part.from_bytes(data=f.read(), mime_type="image/jpeg")
                        )
                else:
                    raise ValueError(
                        "Image content must have either 'url', 'image', or 'image_path'."
                    )
            elif content_part.get("type") == "audio":
                if content_part.get("url") is not None:
                    data.parts.append(
                        types.Part.from_uri(file_uri=content_part.get("url"))
                    )
                elif content_part.get("file_path") is not None:
                    with open(content_part.get("file_path"), "rb") as f:
                        data.parts.append(
                            types.Part.from_bytes(data=f.read(), mime_type="audio/mpeg")
                        )
                elif content_part.get("binary") is not None:
                    data.parts.append(
                        types.Part.from_bytes(
                            data=content_part.get("binary"), mime_type="audio/mpeg"
                        )
                    )
                else:
                    raise ValueError(
                        "Audio content must have either 'url', 'file_path', or 'binary'."
                    )
            elif content_part.get("type") == "video":
                if content_part.get("url") is not None:
                    data.parts.append(
                        types.Part.from_uri(file_uri=content_part.get("url"))
                    )
                elif content_part.get("file_path") is not None:
                    with open(content_part.get("file_path"), "rb") as f:
                        data.parts.append(
                            types.Part.from_bytes(data=f.read(), mime_type="video/mp4")
                        )
                elif content_part.get("binary") is not None:
                    data.parts.append(
                        types.Part.from_bytes(
                            data=content_part.get("binary"), mime_type="video/mp4"
                        )
                    )
                else:
                    raise ValueError(
                        "Video content must have either 'url', 'file_path', or 'binary'."
                    )
            elif content_part.get("type") == "pdf":
                if content_part.get("url") is not None:
                    data.parts.append(
                        types.Part.from_uri(file_uri=content_part.get("url"))
                    )
                elif content_part.get("file_path") is not None:
                    with open(content_part.get("file_path"), "rb") as f:
                        data.parts.append(
                            types.Part.from_bytes(
                                data=f.read(), mime_type="application/pdf"
                            )
                        )
                elif content_part.get("binary") is not None:
                    data.parts.append(
                        types.Part.from_bytes(
                            data=content_part.get("binary"), mime_type="application/pdf"
                        )
                    )
                else:
                    raise ValueError(
                        "PDF content must have either 'url', 'file_path', or 'binary'."
                    )
            else:
                raise ValueError(
                    f"Unsupported content type: {content_part.get('type')}"
                )
        return data

    @staticmethod
    def _google_to_turn(data) -> ChatTurn:
        content = []
        for part in data.parts:
            if part.text is not None:
                content.append({"type": "text", "text": part.text})
            elif part.inline_data is not None:
                content.append({"type": "image", "image": part.as_image()})
        return ChatTurn(role="assistant", content=content)
