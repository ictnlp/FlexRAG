import numpy as np

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
        assert response[0][0].text_content == "Mocked LiteLLM chat response"
        call = mock_litellm_client["calls"]["acompletion"][0]
        assert call["model"] == "openai/gpt-4o-mini"
        assert call["messages"][0]["content"] == "Who is Bruce Wayne?"
        mock_litellm_client["module"].completion.assert_not_called()

    def test_chat_multimodal_payload(self, mock_litellm_client, tmp_path):
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 sample")
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
        assert content[2]["file"]["filename"] == "sample.mp3"
        assert content[3]["file"]["filename"] == "sample.mp4"
        assert content[1]["file"]["file_data"].startswith(
            "data:application/pdf;base64,"
        )

    def test_generate_uses_text_completion(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        response = generator.generate(
            "The capital of China is",
            GenerationConfig(do_sample=False, max_new_tokens=16, stop_str=["."]),
        )
        assert response[0][0] == "Mocked LiteLLM text completion"
        assert len(mock_litellm_client["calls"]["acompletion"]) == 0
        call = mock_litellm_client["calls"]["atext_completion"][0]
        assert call["model"] == "openai/gpt-4o-mini"
        assert call["prompt"] == "The capital of China is"
        assert call["kwargs"]["max_tokens"] == 16
        assert call["kwargs"]["stop"] == ["."]
        assert call["kwargs"]["temperature"] == 0.0
        mock_litellm_client["module"].text_completion.assert_not_called()

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
        assert response.response.text_content == "Mocked LiteLLM chat response"
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
