from typing import Optional

from flexrag.common import TIME_METER, ChatMessages, ChatTurn, configure
from flexrag.common.base64_utils import file_to_base64, image_to_base64, url_to_base64

from .generator_base import GENERATORS, GenerationConfig
from .remote_generator_base import RemoteGeneratorBase, RemoteGeneratorBaseConfig


@configure
class OllamaGeneratorConfig(RemoteGeneratorBaseConfig):
    """Configuration for the OllamaGenerator.

    :param model_name: The name of the model to use. Required.
    :type model_name: str
    :param base_url: The base URL of the Ollama server.
        Default is 'http://localhost:11434/'.
    :type base_url: str
    :param max_concurrency: The maximum number of concurrent generation requests. Default is 1.
    :type max_concurrency: int
    """

    model_name: Optional[str] = None
    base_url: str = "http://localhost:11434/"
    verbose: bool = False


@GENERATORS("ollama", config_class=OllamaGeneratorConfig)
class OllamaGenerator(RemoteGeneratorBase):
    async def _create_client(self, config: OllamaGeneratorConfig):
        from ollama import AsyncClient

        self._model_name = config.model_name
        return AsyncClient(host=config.base_url)

    @TIME_METER("generator.ollama_generate")
    async def _async_chat_impl(
        self,
        client,
        message: ChatMessages,
        generation_config: GenerationConfig | None,
    ) -> str:
        gen_cfg = self._get_options(generation_config)
        message = [self._turn_to_ollama(turn) for turn in message]
        resp = await client.chat(messages=message, **gen_cfg)
        return self._ollama_to_turn(resp)

    @TIME_METER("generator.ollama_generate")
    async def _async_generate_impl(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig | None,
    ) -> str:
        gen_cfg = self._get_options(generation_config)
        resp = await client.generate(prompt=prompt, raw=True, **gen_cfg)
        return resp.response

    def _get_options(self, generation_config: GenerationConfig | None) -> dict:
        if generation_config is None:
            generation_config = GenerationConfig()
        options = {
            "top_k": generation_config.top_k,
            "top_p": generation_config.top_p,
            "stop": list(generation_config.stop_str),
        }
        if generation_config.max_new_tokens is not None:
            options["num_predict"] = generation_config.max_new_tokens
        if generation_config.do_sample:
            options["temperature"] = generation_config.temperature
        else:
            options["temperature"] = 0.0
        return {"model": self._model_name, "options": options}

    @staticmethod
    def _turn_to_ollama(turn: ChatTurn) -> dict:
        if isinstance(turn.content, str):
            return {"role": turn.role, "content": turn.content}
        data = {"role": turn.role, "content": []}
        for content_part in turn.content:
            if content_part.get("type") == "text":
                data["content"] = content_part.get("text", "")
            elif content_part.get("type") == "image":
                if "images" not in data:
                    data["images"] = []
                if content_part.get("url") is not None:
                    base64_image = url_to_base64(content_part.get("url"))
                    data["images"].append(base64_image)
                elif content_part.get("image") is not None:
                    base64_image = image_to_base64(
                        content_part.get("image"), format="JPEG"
                    )
                    data["images"].append(base64_image)
                elif content_part.get("image_path") is not None:
                    base64_image = file_to_base64(content_part.get("image_path"))
                    data["images"].append(base64_image)
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
    def _ollama_to_turn(data: dict) -> ChatTurn:
        return ChatTurn(role="assistant", content=data.message.content)
