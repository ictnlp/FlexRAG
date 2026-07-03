import re

import numpy as np
import pytest
from PIL import Image

from flexrag.common import LOGGER_MANAGER
from flexrag.common.dataclasses import ChatMessages, ChatTurn
from flexrag.models import (
    GenerationConfig,
    HFClipEncoder,
    HFClipEncoderConfig,
    HFEncoder,
    HFEncoderConfig,
    HFGenerator,
    HFGeneratorConfig,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
)
from flexrag.models.encoders import EncoderProtocol
from flexrag.models.generators import GeneratorProtocol

pytestmark = pytest.mark.integration

logger = LOGGER_MANAGER.get_logger("tests.test_model")


class TestGenerator:
    prompts = [
        ChatMessages(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")]),
        ChatMessages(history=[ChatTurn(role="user", content="Who is Thomas Wayne?")]),
        ChatMessages(history=[ChatTurn(role="user", content="What is the capital of China?")]),
    ]  # fmt: skip
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

    async def valid_chat_function(self, generator: GeneratorProtocol):
        # test chat & async_chat with sampling
        r1 = generator.chat(self.prompts, self.sampled_cfg, batch_size=2)
        self.valid_chat_sampled(r1)
        r2 = await generator.async_chat(self.prompts, self.sampled_cfg, batch_size=2)
        self.valid_chat_sampled(r2)
        # test chat & async_chat with default generation config
        r1 = generator.chat(self.prompts, batch_size=2)
        self.valid_chat_default(r1)
        r2 = await generator.async_chat(self.prompts, batch_size=2)
        self.valid_chat_default(r2)
        # test chat & async_chat with stop strings
        r1 = generator.chat(self.prompts, self.stopped_cfg, batch_size=2)
        self.valid_chat_stopped(r1)
        r2 = await generator.async_chat(self.prompts, self.stopped_cfg, batch_size=2)
        self.valid_chat_stopped(r2)
        return

    async def valid_generate_function(self, generator: GeneratorProtocol):
        # test generate & async_generate with sampling
        r1 = generator.generate(self.prefixes, self.sampled_cfg, batch_size=2)
        self.valid_sampled(r1)
        r2 = await generator.async_generate(
            self.prefixes, self.sampled_cfg, batch_size=2
        )
        self.valid_sampled(r2)
        # test generate & async_generate with default generation config
        r1 = generator.generate(self.prefixes, batch_size=2)
        self.valid_default(r1)
        r2 = await generator.async_generate(self.prefixes, batch_size=2)
        self.valid_default(r2)
        # test generate & async_generate with stop strings
        r1 = generator.generate(self.prefixes, self.stopped_cfg, batch_size=2)
        self.valid_stopped(r1)
        r2 = await generator.async_generate(
            self.prefixes, self.stopped_cfg, batch_size=2
        )
        self.valid_stopped(r2)
        return

    async def run_generator(self, generator: GeneratorProtocol):
        try:
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
        finally:
            close = getattr(generator, "close", None)
            if callable(close):
                close()
        return

    @pytest.mark.gpu
    @pytest.mark.asyncio
    async def test_hf(self):
        # if not torch.cuda.is_available():
        #     pytest.skip("HFGenerator requires GPU for inference")
        generator = HFGenerator(
            HFGeneratorConfig(
                model_path="Qwen/Qwen3-0.6B",
                device_map=0,
            )
        )
        await self.run_generator(generator)
        return


class TestEncode:
    text = [
        "Who is Bruce Wayne?",
        "Who is Thomas Wayne?",
        "What is the capital of China?",
    ]

    async def run_encoder(self, encoder: EncoderProtocol) -> None:
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
    async def test_hf(self):
        encoder = HFEncoder(HFEncoderConfig(model_path="facebook/contriever"))
        await self.run_encoder(encoder)
        return

    @pytest.mark.asyncio
    async def test_hf_text_content_parts(self):
        encoder = HFEncoder(HFEncoderConfig(model_path="facebook/contriever"))
        try:
            inputs = [{"type": "text", "text": text} for text in self.text]
            embeddings = encoder.encode(inputs)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == len(self.text)
        finally:
            encoder.close()

    @pytest.mark.asyncio
    async def test_hf_rejects_non_text_content_parts(self):
        encoder = HFEncoder(HFEncoderConfig(model_path="facebook/contriever"))
        try:
            with pytest.raises(RuntimeError, match="only supports text content blocks"):
                encoder.encode([{"type": "image", "image_path": "/tmp/clip.png"}])
        finally:
            encoder.close()

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
    async def test_sentence_transformer_text_content_parts(self):
        encoder = SentenceTransformerEncoder(
            SentenceTransformerEncoderConfig(
                model_path="sentence-transformers/all-MiniLM-L6-v2",
            )
        )
        try:
            inputs = [{"type": "text", "text": text} for text in self.text]
            embeddings = encoder.encode(inputs)
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == len(self.text)
        finally:
            encoder.close()

    @pytest.mark.asyncio
    async def test_hf_clip(self, tmp_path):
        image_path = tmp_path / "clip-image.png"
        Image.new("RGB", (4, 4), color="red").save(image_path)
        encoder = HFClipEncoder(
            HFClipEncoderConfig(model_path="openai/clip-vit-base-patch32")
        )
        try:
            text_embeddings = encoder.encode(self.text)
            assert isinstance(text_embeddings, np.ndarray)
            assert text_embeddings.shape[0] == len(self.text)
            assert text_embeddings.shape[1] == encoder.embedding_size

            async_text_embeddings = await encoder.async_encode(self.text)
            assert np.allclose(text_embeddings, async_text_embeddings)

            mixed_inputs = [
                {"type": "text", "text": self.text[0]},
                {"type": "image", "image_path": str(image_path)},
                {"type": "text", "text": self.text[1]},
            ]
            mixed_embeddings = encoder.encode(mixed_inputs)
            async_mixed_embeddings = await encoder.async_encode(mixed_inputs)
            assert mixed_embeddings.shape == (3, encoder.embedding_size)
            assert np.allclose(mixed_embeddings, async_mixed_embeddings)

            reference_text = encoder.encode([self.text[0], self.text[1]])
            reference_image = encoder.encode(
                [{"type": "image", "image_path": str(image_path)}]
            )
            assert np.allclose(mixed_embeddings[0], reference_text[0])
            assert np.allclose(mixed_embeddings[1], reference_image[0])
            assert np.allclose(mixed_embeddings[2], reference_text[1])
        finally:
            encoder.close()
        return
