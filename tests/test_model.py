import re

import numpy as np
import pytest

from flexrag.common import LOGGER_MANAGER
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models import (
    EncoderBase,
    GenerationConfig,
    GeneratorBase,
    HFClipEncoder,
    HFClipEncoderConfig,
    HFEncoder,
    HFEncoderConfig,
    HFGenerator,
    HFGeneratorConfig,
    LiteLLMEncoder,
    LiteLLMEncoderConfig,
    LiteLLMGenerator,
    LiteLLMGeneratorConfig,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)

logger = LOGGER_MANAGER.get_logger("tests.test_model")


class TestGenerator:
    prompts = [
        ChatMessages(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")]),
        ChatMessages(history=[ChatTurn(role="user", content="Who is Thomas Wayne?")]),
        ChatMessages(history=[ChatTurn(role="user", content="What is the capital of China?")]),  # fmt: skip
    ]
    prefixes = [
        "Bruce Wayne is A comic book superhero",
        "Thomas Wayne is the father",
        "The capital of China is",
    ]
    sampled_cfg = GenerationConfig(
        do_sample=True,
        sample_num=3,
        temperature=0.7,
        max_new_tokens=50,
    )
    stopped_cfg = GenerationConfig(stop_str=["."], max_new_tokens=100)

    def valid_sampled(self, results: list[list[str]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == self.sampled_cfg.sample_num
            for rr in r:
                assert isinstance(rr, str)
        return

    def valid_stopped(self, results: list[list[str]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == 1
            assert isinstance(r[0], str)
            assert len(re.findall(r"\.", r[0])) <= 1
        return

    def valid_default(self, results: list[list[str]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == 1
            assert isinstance(r[0], str)
        return

    def valid_chat_sampled(self, results: list[list[ChatTurn]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == self.sampled_cfg.sample_num
            for rr in r:
                assert isinstance(rr, ChatTurn)
                assert isinstance(rr.text_content, str)
        return

    def valid_chat_stopped(self, results: list[list[ChatTurn]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == 1
            assert isinstance(r[0], ChatTurn)
            assert isinstance(r[0].text_content, str)
            assert len(re.findall(r"\.", r[0].text_content)) <= 1
        return

    def valid_chat_default(self, results: list[list[ChatTurn]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == 1
            assert isinstance(r[0], ChatTurn)
            assert isinstance(r[0].text_content, str)
        return

    async def valid_chat_function(self, generator: GeneratorBase):
        # test chat & async_chat with sampling
        r1 = generator.chat(self.prompts, self.sampled_cfg)
        self.valid_chat_sampled(r1)
        r2 = await generator.async_chat(self.prompts, self.sampled_cfg)
        self.valid_chat_sampled(r2)
        # test chat & async_chat with default generation config
        r1 = generator.chat(self.prompts)
        self.valid_chat_default(r1)
        r2 = await generator.async_chat(self.prompts)
        self.valid_chat_default(r2)
        # test chat & async_chat with stop strings
        r1 = generator.chat(self.prompts, self.stopped_cfg)
        self.valid_chat_stopped(r1)
        r2 = await generator.async_chat(self.prompts, self.stopped_cfg)
        self.valid_chat_stopped(r2)
        return

    async def valid_generate_function(self, generator: GeneratorBase):
        # test generate & async_generate with sampling
        r1 = generator.generate(self.prefixes, self.sampled_cfg)
        self.valid_sampled(r1)
        r2 = await generator.async_generate(self.prefixes, self.sampled_cfg)
        self.valid_sampled(r2)
        # test generate & async_generate with default generation config
        r1 = generator.generate(self.prefixes)
        self.valid_default(r1)
        r2 = await generator.async_generate(self.prefixes)
        self.valid_default(r2)
        # test generate & async_generate with stop strings
        r1 = generator.generate(self.prefixes, self.stopped_cfg)
        self.valid_stopped(r1)
        r2 = await generator.async_generate(self.prefixes, self.stopped_cfg)
        self.valid_stopped(r2)
        return

    @pytest.mark.asyncio
    async def test_litellm_openai(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="openai", model_name="gpt-4o-mini")
        )
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return

    @pytest.mark.asyncio
    async def test_litellm_ollama(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="ollama_chat", model_name="llama3.1")
        )
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_vllm(self):
        # FIXME: VLLMGenerator can not be initiate if CUDA is initialized.
        # if not torch.cuda.is_available():
        #     pytest.skip("VLLMGenerator requires GPU for inference")
        generator = VLLMGenerator(VLLMGeneratorConfig(model_path="Qwen/Qwen3-0.6B"))
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_hf(self):
        # if not torch.cuda.is_available():
        #     pytest.skip("HFGenerator requires GPU for inference")
        generator = HFGenerator(
            HFGeneratorConfig(
                model_path="Qwen/Qwen3-0.6B",
                device_id=[0],
            )
        )
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return

    @pytest.mark.asyncio
    async def test_litellm_anthropic(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="anthropic", model_name="claude-3-7-sonnet")
        )
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return

    @pytest.mark.asyncio
    async def test_litellm_gemini(self, mock_litellm_client):
        generator = LiteLLMGenerator(
            LiteLLMGeneratorConfig(provider="gemini", model_name="gemini-2.0-flash")
        )
        await self.valid_chat_function(generator)
        await self.valid_generate_function(generator)
        return


class TestEncode:
    text = [
        "Who is Bruce Wayne?",
        "Who is Thomas Wayne?",
        "What is the capital of China?",
    ]

    async def run_encoder(self, encoder: EncoderBase) -> None:
        try:
            r1 = encoder.encode(self.text)
            assert isinstance(r1, np.ndarray)
            assert r1.ndim == 2
            assert r1.shape[0] == len(self.text)
            assert r1.shape[1] == encoder.embedding_size
            r2 = await encoder.async_encode(self.text)
            assert r1.ndim == 2
            assert r1.shape[0] == len(self.text)
            assert r1.shape[1] == encoder.embedding_size
            assert (r1 - r2).max() < 1e-4
        finally:
            close = getattr(encoder, "close", None)
            if callable(close):
                close()

    @pytest.mark.asyncio
    async def test_litellm_openai(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="openai",
                model_name="text-embedding-3-small",
                embedding_size=512,
            )
        )
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_litellm_ollama(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="ollama",
                model_name="contriever",
                embedding_size=768,
            )
        )
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_hf(self):
        encoder = HFEncoder(HFEncoderConfig(model_path="facebook/contriever"))
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_sentence_transformer(self):
        encoder = SentenceTransformerEncoder(
            SentenceTransformerEncoderConfig(
                model_path="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_hf_clip(self):
        # test openai clip model
        encoder = HFClipEncoder(
            HFClipEncoderConfig(model_path="openai/clip-vit-base-patch32")
        )
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_litellm_jina(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="jina_ai",
                model_name="jina-embeddings-v3",
                embedding_size=768,
            )
        )
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_litellm_cohere(self, mock_litellm_client):
        encoder = LiteLLMEncoder(
            LiteLLMEncoderConfig(
                provider="cohere",
                model_name="embed-v4.0",
                embedding_size=1536,
                input_type="search_document",
            )
        )
        await self.run_encoder(encoder)
        return
