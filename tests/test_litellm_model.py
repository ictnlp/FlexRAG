import types

import numpy as np
import pytest
from aiohttp import ClientSession
from PIL import Image

from flexrag.assistants import ModularAssistant, ModularAssistantConfig
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models import (
    ENCODERS,
    GENERATORS,
    EncoderConfig,
    GenerationConfig,
    GeneratorConfig,
    LiteLLMEncoder,
    LiteLLMEncoderConfig,
    LiteLLMGenerator,
    LiteLLMGeneratorConfig,
)
from flexrag.processors.chunkers import SemanticChunker, SemanticChunkerConfig


class TestLiteLLMGenerator:
    def test_generator_config_union(self):
        cfg = GeneratorConfig(
            generator_type="litellm",
            litellm_config=LiteLLMGeneratorConfig(
                provider="openai",
                model_name="gpt-4o-mini",
            ),
        )
        generator = GENERATORS.load(cfg)
        assert isinstance(generator, LiteLLMGenerator)

    def test_chat_text(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        prompts = [
            ChatMessages(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")])
        ]
        response = generator.chat(prompts, GenerationConfig(do_sample=False))
        assert response[0][0].text_content == "Mocked LiteLLM chat response 0"
        call = mock_litellm_client["calls"]["acompletion"][0]
        assert call["model"] == "openai/gpt-4o-mini"
        assert call["messages"][0]["content"] == "Who is Bruce Wayne?"
        mock_litellm_client["module"].completion.assert_not_called()

    def test_owned_session_is_forwarded_and_closed(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(
                provider="anthropic",
                model_name="claude-sonnet-4-20250514",
            )
        )
        generator.chat([ChatMessages(history=[ChatTurn(role="user", content="Ping")])])

        call = mock_litellm_client["calls"]["acompletion"][0]
        session = call["kwargs"]["shared_session"]
        assert isinstance(session, ClientSession)
        assert not session.closed

        generator.close()
        assert session.closed

    def test_borrowed_session_is_not_closed(self, mocker, mock_litellm_client):
        session = mocker.MagicMock()
        session.close = mocker.AsyncMock()
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                extra_kwargs={"shared_session": session},
            )
        )
        generator.chat([ChatMessages(history=[ChatTurn(role="user", content="Ping")])])

        call = mock_litellm_client["calls"]["acompletion"][0]
        assert call["kwargs"]["shared_session"] is session

        generator.close()
        session.close.assert_not_awaited()

    def test_chat_multimodal_payload(self, mock_litellm_client, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 sample")
        html_path = tmp_path / "sample.html"
        html_path.write_text("<html><body>sample</body></html>", encoding="utf-8")
        audio_path = tmp_path / "sample.mp3"
        audio_path.write_bytes(b"mp3")
        video_path = tmp_path / "sample.mp4"
        video_path.write_bytes(b"mp4")

        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        messages = [
            ChatMessages(
                history=[
                    ChatTurn(
                        role="user",
                        content=[
                            {"type": "text", "text": "Describe these files"},
                            {"type": "pdf", "file_path": str(pdf_path)},
                            {
                                "type": "file",
                                "file_path": str(html_path),
                                "mime_type": "text/html",
                                "file_name": "sample.html",
                            },
                            {"type": "audio", "file_path": str(audio_path)},
                            {"type": "video", "file_path": str(video_path)},
                        ],
                    )
                ]
            )
        ]
        generator.chat(messages)
        content = mock_litellm_client["calls"]["acompletion"][0]["messages"][0][
            "content"
        ]
        assert content[0] == {"type": "text", "text": "Describe these files"}
        assert content[1]["type"] == "file"
        assert content[1]["file"]["filename"] == "sample.pdf"
        assert content[2]["file"]["filename"] == "sample.html"
        assert content[3]["file"]["filename"] == "sample.mp3"
        assert content[4]["file"]["filename"] == "sample.mp4"
        assert content[1]["file"]["file_data"].startswith(
            "data:application/pdf;base64,"
        )
        assert content[2]["file"]["file_data"].startswith("data:text/html;base64,")

    def test_generate_uses_text_completion(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        response = generator.generate(
            "The capital of China is",
            GenerationConfig(do_sample=False, max_new_tokens=16, stop_str=["."]),
        )
        assert response[0][0] == "Mocked LiteLLM text completion 0"
        assert len(mock_litellm_client["calls"]["acompletion"]) == 0
        call = mock_litellm_client["calls"]["atext_completion"][0]
        assert call["model"] == "openai/gpt-4o-mini"
        assert call["prompt"] == "The capital of China is"
        assert call["kwargs"]["max_tokens"] == 16
        assert call["kwargs"]["stop"] == ["."]
        assert call["kwargs"]["temperature"] == 0.0
        assert call["kwargs"]["n"] == 1
        mock_litellm_client["module"].text_completion.assert_not_called()

    def test_generate_sampled_returns_multiple_choices(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        response = generator.generate(
            "The capital of China is",
            GenerationConfig(do_sample=True, sample_num=3, max_new_tokens=16),
        )
        assert response == [[f"Mocked LiteLLM text completion {i}" for i in range(3)]]
        call = mock_litellm_client["calls"]["atext_completion"][0]
        assert call["kwargs"]["n"] == 3

    def test_extra_kwargs_passthrough_with_explicit_precedence(
        self, mock_litellm_client
    ):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(
                provider="openai",
                model_name="gpt-4o-mini",
                timeout=30.0,
                extra_kwargs={
                    "timeout": 5.0,
                    "custom_llm_provider": "openai",
                    "metadata": {"source": "test"},
                },
            )
        )
        generator.chat(
            [ChatMessages(history=[ChatTurn(role="user", content="Ping")])],
            GenerationConfig(do_sample=False),
        )
        call = mock_litellm_client["calls"]["acompletion"][0]
        assert call["kwargs"]["timeout"] == 30.0
        assert call["kwargs"]["custom_llm_provider"] == "openai"
        assert call["kwargs"]["metadata"] == {"source": "test"}

    def test_chat_tool_calls_and_metadata(self, mock_litellm_client):
        async def mock_tool_call_response(*, model, messages, **kwargs):
            mock_litellm_client["calls"]["acompletion"].append(
                {"model": model, "messages": messages, "kwargs": kwargs}
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        finish_reason="tool_calls",
                        message=types.SimpleNamespace(
                            role="assistant",
                            content=[{"type": "text", "text": "Checking the weather."}],
                            tool_calls=[
                                types.SimpleNamespace(
                                    id="call_1",
                                    function=types.SimpleNamespace(
                                        name="get_weather",
                                        arguments='{"city":"Beijing"}',
                                    ),
                                )
                            ],
                            reasoning_content=None,
                            thinking_blocks=None,
                        ),
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                ),
            )

        mock_litellm_client["module"].acompletion.side_effect = mock_tool_call_response
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        response = generator.chat(
            [ChatMessages(history=[ChatTurn(role="user", content="Weather?")])]
        )

        turn = response[0][0]
        assert turn.text_content == "Checking the weather."
        assert bool(turn.tool_calls)
        assert turn.tool_calls[0]["name"] == "get_weather"
        assert turn.tool_calls[0]["arguments"] == {"city": "Beijing"}
        assert turn.metadata["finish_reason"] == "tool_calls"
        assert turn.metadata["usage"]["total_tokens"] == 18

    def test_chat_reasoning_fields_are_normalized(self, mock_litellm_client):
        async def mock_reasoning_response(*, model, messages, **kwargs):
            mock_litellm_client["calls"]["acompletion"].append(
                {"model": model, "messages": messages, "kwargs": kwargs}
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        finish_reason="stop",
                        message=types.SimpleNamespace(
                            role="assistant",
                            content="The answer is Paris.",
                            tool_calls=None,
                            reasoning_content="Need to reason first.",
                            thinking_blocks=[
                                {
                                    "type": "thinking",
                                    "thinking": "Need to reason first.",
                                }
                            ],
                        ),
                    )
                ],
                usage=types.SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=7,
                    total_tokens=18,
                ),
            )

        mock_litellm_client["module"].acompletion.side_effect = mock_reasoning_response
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        response = generator.chat(
            [
                ChatMessages(
                    history=[ChatTurn(role="user", content="Capital of France?")]
                )
            ]
        )

        turn = response[0][0]
        assert turn.content == "The answer is Paris."
        assert turn.reasoning_content == "Need to reason first."
        assert turn.thinking_blocks == [
            {"type": "thinking", "thinking": "Need to reason first."}
        ]
        assert turn.metadata["finish_reason"] == "stop"
        assert "reasoning_content" not in turn.metadata
        assert "thinking_blocks" not in turn.metadata

    def test_tool_result_message_includes_tool_call_id(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        messages = ChatMessages(
            history=[
                ChatTurn(
                    role="user",
                    content="Please add 32767.5 and 65537.5.",
                ),
                ChatTurn(
                    role="assistant",
                    content=[
                        {
                            "type": "tool_call",
                            "id": "call_1",
                            "name": "add",
                            "arguments": {"a": 32767.5, "b": 65537.5},
                        }
                    ],
                ),
                ChatTurn(
                    role="tool",
                    tool_call_id="call_1",
                    name="add",
                    content="98305.0",
                ),
            ]
        )

        generator.chat([messages])
        serialized_messages = mock_litellm_client["calls"]["acompletion"][0]["messages"]
        assert serialized_messages[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "add",
            "content": "98305.0",
        }

    def test_chat_passes_tools_and_reasoning_effort(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather by city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                },
            }
        ]
        generator.chat(
            [ChatMessages(history=[ChatTurn(role="user", content="Weather?")])],
            GenerationConfig(
                do_sample=False,
                tools=tools,
                reasoning_effort="high",
            ),
        )

        call = mock_litellm_client["calls"]["acompletion"][0]
        assert call["kwargs"]["tools"] == tools
        assert call["kwargs"]["reasoning_effort"] == "high"

    def test_generate_ignores_tools_and_reasoning_effort(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        generator.generate(
            "Weather in Beijing is",
            GenerationConfig(
                do_sample=False,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                reasoning_effort="high",
            ),
        )

        call = mock_litellm_client["calls"]["atext_completion"][0]
        assert "tools" not in call["kwargs"]
        assert "reasoning_effort" not in call["kwargs"]


class TestLiteLLMEncoder:
    def test_encoder_config_union(self):
        cfg = EncoderConfig(
            encoder_type="litellm",
            litellm_config=LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
            ),
        )
        encoder = ENCODERS.load(cfg)
        assert isinstance(encoder, LiteLLMEncoder)

    def test_encode(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
                input_type="search_document",
            )
        )
        embeddings = encoder.encode(["Who is Bruce Wayne?", "Who is Thomas Wayne?"])
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, 8)
        call = mock_litellm_client["calls"]["aembedding"][0]
        assert call["model"] == "openai/text-embedding-3-small"
        assert call["input_type"] == "search_document"
        assert call["dimensions"] == 8
        mock_litellm_client["module"].embedding.assert_not_called()

    def test_encoder_owned_session_is_forwarded_and_closed(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="cohere",
                model_name="embed-v4.0",
            )
        )
        encoder.encode(["Who is Bruce Wayne?"])

        call = mock_litellm_client["calls"]["aembedding"][0]
        session = call["kwargs"]["shared_session"]
        assert isinstance(session, ClientSession)
        assert not session.closed

        encoder.close()
        assert session.closed

    def test_encode_extra_kwargs_passthrough_with_explicit_precedence(
        self, mock_litellm_client
    ):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                timeout=30.0,
                input_type="search_document",
                extra_kwargs={
                    "timeout": 5.0,
                    "metadata": {"source": "test"},
                    "dimensions": 99,
                },
            )
        )
        encoder.encode(["Who is Bruce Wayne?"])
        call = mock_litellm_client["calls"]["aembedding"][0]
        assert call["kwargs"]["timeout"] == 30.0
        assert call["kwargs"]["metadata"] == {"source": "test"}
        assert call["dimensions"] is None
        assert call["input_type"] == "search_document"

    def test_encode_image_url(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
            )
        )
        embeddings = encoder.encode(
            [{"type": "image", "url": "https://example.com/a.png"}]
        )
        assert embeddings.shape == (1, 8)
        call = mock_litellm_client["calls"]["aembedding"][0]
        assert call["input"] == [
            [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]
        ]

    def test_encode_image_from_path_and_memory(self, mock_litellm_client, tmp_path):
        image_path = tmp_path / "sample.png"
        Image.new("RGB", (4, 4), color="red").save(image_path)
        memory_image = Image.open(image_path)

        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
            )
        )
        embeddings = encoder.encode(
            [
                {"type": "image", "image_path": str(image_path)},
                {"type": "image", "image": memory_image},
            ]
        )
        assert embeddings.shape == (2, 8)

        call = mock_litellm_client["calls"]["aembedding"][0]
        first_url = call["input"][0][0]["image_url"]["url"]
        second_url = call["input"][1][0]["image_url"]["url"]
        assert first_url.startswith("data:image/png;base64,")
        assert second_url.startswith("data:image/jpeg;base64,")

    def test_encode_mixed_text_image_batch(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
            )
        )
        mixed_embeddings = encoder.encode(
            [
                "Who is Bruce Wayne?",
                {"type": "image", "url": "https://example.com/a.png"},
                {"type": "text", "text": "Who is Thomas Wayne?"},
            ]
        )
        assert mixed_embeddings.shape == (3, 8)
        assert len(mock_litellm_client["calls"]["aembedding"]) == 2

        reference_text = encoder.encode(["Who is Bruce Wayne?", "Who is Thomas Wayne?"])
        reference_image = encoder.encode(
            [{"type": "image", "url": "https://example.com/a.png"}]
        )
        assert np.allclose(mixed_embeddings[0], reference_text[0])
        assert np.allclose(mixed_embeddings[1], reference_image[0])
        assert np.allclose(mixed_embeddings[2], reference_text[1])

    def test_encode_rejects_non_image_multimodal_parts(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=8,
            )
        )
        with pytest.raises(
            ValueError,
            match="LiteLLMEncoder only supports text and image content blocks",
        ):
            encoder.encode([{"type": "audio", "url": "https://example.com/a.mp3"}])


class TestLiteLLMIntegration:
    def test_modular_assistant_with_litellm(self, mock_litellm_client):
        assistant = ModularAssistant(
            ModularAssistantConfig(
                generator_type="litellm",
                litellm_config=LiteLLMGeneratorConfig(
                    provider="openai",
                    model_name="gpt-4o-mini",
                ),
            )
        )
        response = assistant.answer(
            ChatMessages(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")])
        )
        assert response.response.text_content == "Mocked LiteLLM chat response 0"
        assert mock_litellm_client["calls"]["acompletion"]

    def test_semantic_chunker_with_litellm(self, mock_litellm_client):
        chunker = SemanticChunker(
            SemanticChunkerConfig(
                threshold_percentile=50,
                encoder_type="litellm",
                litellm_config=LiteLLMEncoderConfig(
                    provider="openai",
                    model_name="text-embedding-3-small",
                    embedding_size=8,
                ),
            )
        )
        chunks = chunker.chunk(
            "This is the first sentence. This is the second sentence.",
            return_str=True,
        )
        assert len(chunks) > 0
        assert mock_litellm_client["calls"]["aembedding"]
