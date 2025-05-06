import os
import re
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from flexrag.models import (
    AnthropicGenerator,
    AnthropicGeneratorConfig,
    CohereEncoder,
    CohereEncoderConfig,
    GenerationConfig,
    GeneratorBase,
    HFClipEncoder,
    HFClipEncoderConfig,
    HFEncoder,
    HFEncoderConfig,
    HFGenerator,
    HFGeneratorConfig,
    JinaEncoder,
    JinaEncoderConfig,
    LlamacppGenerator,
    LlamacppGeneratorConfig,
    OllamaEncoder,
    OllamaEncoderConfig,
    OllamaGenerator,
    OllamaGeneratorConfig,
    OpenAIEncoder,
    OpenAIEncoderConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.utils import LOGGER_MANAGER, ConfigureBase

logger = LOGGER_MANAGER.get_logger("tests.test_model")


@dataclass
class GeneratorTestConfig(ConfigureBase):
    openai_configs: list[OpenAIGeneratorConfig] = field(default_factory=list)
    ollama_configs: list[OllamaGeneratorConfig] = field(default_factory=list)
    anthropic_configs: list[AnthropicGeneratorConfig] = field(default_factory=list)
    vllm_configs: list[VLLMGeneratorConfig] = field(default_factory=list)
    hf_configs: list[HFGeneratorConfig] = field(default_factory=list)
    llamacpp_configs: list[LlamacppGeneratorConfig] = field(default_factory=list)


class TestGenerator:
    cfg: GeneratorTestConfig = OmegaConf.merge(
        OmegaConf.structured(GeneratorTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "generator.yaml")
        ),
    )
    prompts = [
        ChatPrompt(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")]),
        ChatPrompt(history=[ChatTurn(role="user", content="Who is Thomas Wayne?")]),
        ChatPrompt(history=[ChatTurn(role="user", content="What is the capital of China?")]),  # fmt: skip
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

    async def valid_chat_function(self, generator: GeneratorBase):
        # test chat & async_chat with sampling
        r1 = generator.chat(self.prompts, self.sampled_cfg)
        self.valid_sampled(r1)
        r2 = await generator.async_chat(self.prompts, self.sampled_cfg)
        self.valid_sampled(r2)
        # test chat & async_chat with default generation config
        r1 = generator.chat(self.prompts)
        self.valid_default(r1)
        r2 = await generator.async_chat(self.prompts)
        self.valid_default(r2)
        # test chat & async_chat with stop strings
        r1 = generator.chat(self.prompts, self.stopped_cfg)
        self.valid_stopped(r1)
        r2 = await generator.async_chat(self.prompts, self.stopped_cfg)
        self.valid_stopped(r2)
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
    async def test_openai(self):
        for cfg in self.cfg.openai_configs:
            logger.debug(f"Testing OpenAIGenerator({cfg.model_name}).")
            generator = OpenAIGenerator(cfg)
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
            logger.debug(f"Testing OpenAIGenerator({cfg.model_name}) done.")
        return

    @pytest.mark.asyncio
    async def test_ollama(self):
        for cfg in self.cfg.ollama_configs:
            logger.debug(f"Testing OllamaGenerator({cfg.model_name}).")
            generator = OllamaGenerator(cfg)
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
            logger.debug(f"Testing OllamaGenerator({cfg.model_name}) done.")
        return

    @pytest.mark.asyncio
    async def test_vllm(self):
        for cfg in self.cfg.vllm_configs:
            logger.debug(f"Testing VLLMGenerator({cfg.model_path}).")
            generator = VLLMGenerator(cfg)
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
            logger.debug(f"Testing VLLMGenerator({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_hf(self):
        for cfg in self.cfg.hf_configs:
            logger.debug(f"Testing HFGenerator({cfg.model_path}).")
            generator = HFGenerator(cfg)
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
            logger.debug(f"Testing HFGenerator({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_llamacpp(self):
        for cfg in self.cfg.llamacpp_configs:
            logger.debug(f"Testing LlamacppGenerator({cfg.model_path}).")
            generator = LlamacppGenerator(cfg)
            await self.valid_chat_function(generator)
            await self.valid_generate_function(generator)
            logger.debug(f"Testing LlamacppGenerator({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_anthropic(self):
        for cfg in self.cfg.anthropic_configs:
            logger.debug(f"Testing AnthropicGenerator({cfg.model_name}).")
            generator = AnthropicGenerator(cfg)
            await self.valid_chat_function(generator)
            logger.debug(f"Testing AnthropicGenerator({cfg.model_name}) done.")
        return


@dataclass
class EncodeTestConfig(ConfigureBase):
    openai_configs: list[OpenAIEncoderConfig] = field(default_factory=list)
    ollama_configs: list[OllamaEncoderConfig] = field(default_factory=list)
    hf_configs: list[HFEncoderConfig] = field(default_factory=list)
    hf_clip_configs: list[HFClipEncoderConfig] = field(default_factory=list)
    jina_configs: list[JinaEncoderConfig] = field(default_factory=list)
    cohere_configs: list[CohereEncoderConfig] = field(default_factory=list)
    st_configs: list[SentenceTransformerEncoderConfig] = field(default_factory=list)


class TestEncode:
    cfg: EncodeTestConfig = OmegaConf.merge(
        OmegaConf.structured(EncodeTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "encoder.yaml")
        ),
    )
    text = [
        "Who is Bruce Wayne?",
        "Who is Thomas Wayne?",
        "What is the capital of China?",
    ]

    @pytest.mark.asyncio
    async def test_openai(self):
        for cfg in self.cfg.openai_configs:
            logger.debug(f"Testing OpenAIEncoder({cfg.model_name}).")
            encoder = OpenAIEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing OpenAIEncoder({cfg.model_name}) done.")
        return

    @pytest.mark.asyncio
    async def test_ollama(self):
        for cfg in self.cfg.ollama_configs:
            logger.debug(f"Testing OllamaEncoder({cfg.model_name}).")
            encoder = OllamaEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing OllamaEncoder({cfg.model_name}) done.")
        return

    @pytest.mark.asyncio
    async def test_hf(self):
        for cfg in self.cfg.hf_configs:
            logger.debug(f"Testing HFEncoder({cfg.model_path}).")
            encoder = HFEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing HFEncoder({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_sentence_transformer(self):
        for cfg in self.cfg.st_configs:
            logger.debug(f"Testing SentenceTransformerEncoder({cfg.model_path}).")
            encoder = SentenceTransformerEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing SentenceTransformerEncoder({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_hf_clip(self):
        for cfg in self.cfg.hf_clip_configs:
            logger.debug(f"Testing HFClipEncoder({cfg.model_path}).")
            encoder = HFClipEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing HFClipEncoder({cfg.model_path}) done.")
        return

    @pytest.mark.asyncio
    async def test_jina(self):
        for cfg in self.cfg.jina_configs:
            logger.debug(f"Testing JinaEncoder({cfg.model}).")
            encoder = JinaEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing JinaEncoder({cfg.model}) done.")
        return

    @pytest.mark.asyncio
    async def test_cohere(self):
        for cfg in self.cfg.cohere_configs:
            logger.debug(f"Testing CohereEncoder({cfg.model}).")
            encoder = CohereEncoder(cfg)
            r1 = encoder.encode(self.text)
            r2 = await encoder.async_encode(self.text)
            assert (r1 - r2).max() < 1e-4
            logger.debug(f"Testing CohereEncoder({cfg.model}) done.")
        return
