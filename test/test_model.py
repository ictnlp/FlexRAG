import os
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

from kylin.models import (
    AnthropicGenerator,
    AnthropicGeneratorConfig,
    CohereEncoder,
    CohereEncoderConfig,
    GenerationConfig,
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
    VLLMGenerator,
    VLLMGeneratorConfig,
)
from kylin.prompt import ChatPrompt, ChatTurn


@dataclass
class GeneratorTestConfig:
    openai_config: OpenAIGeneratorConfig = field(default_factory=OpenAIGeneratorConfig)
    ollama_config: OllamaGeneratorConfig = field(default_factory=OllamaGeneratorConfig)
    anthropic_config: AnthropicGeneratorConfig = field(default_factory=AnthropicGeneratorConfig)  # fmt: skip
    vllm_config: VLLMGeneratorConfig = field(default_factory=VLLMGeneratorConfig)
    hf_config: HFGeneratorConfig = field(default_factory=HFGeneratorConfig)
    llamacpp_config: LlamacppGeneratorConfig = field(default_factory=LlamacppGeneratorConfig)  # fmt: skip


class TestGenerator:
    cfg: GeneratorTestConfig = OmegaConf.merge(
        OmegaConf.structured(GeneratorTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "generator.yaml")
        ),
    )
    prompts = [
        ChatPrompt(history=[ChatTurn(role="user", content="Who is Bruce Wayne?")])
    ] * 2
    gen_cfg = GenerationConfig(
        do_sample=True,
        sample_num=3,
        temperature=0.7,
    )

    def valid_result(self, results: list[list[str]]) -> None:
        assert len(results) == len(self.prompts)
        for r in results:
            assert len(r) == self.gen_cfg.sample_num
            for rr in r:
                assert isinstance(rr, str)
        return

    @pytest.mark.asyncio
    async def test_openai(self):
        generator = OpenAIGenerator(self.cfg.openai_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return

    @pytest.mark.asyncio
    async def test_ollama(self):
        generator = OllamaGenerator(self.cfg.ollama_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return

    @pytest.mark.asyncio
    async def test_vllm(self):
        generator = VLLMGenerator(self.cfg.vllm_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return

    @pytest.mark.asyncio
    async def test_hf(self):
        generator = HFGenerator(self.cfg.hf_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return

    @pytest.mark.asyncio
    async def test_llamacpp(self):
        generator = LlamacppGenerator(self.cfg.llamacpp_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return

    @pytest.mark.asyncio
    async def test_anthropic(self):
        generator = AnthropicGenerator(self.cfg.anthropic_config)
        r1 = generator.chat(self.prompts, self.gen_cfg)
        self.valid_result(r1)
        r2 = await generator.async_chat(self.prompts, self.gen_cfg)
        self.valid_result(r2)
        return


@dataclass
class EncodeTestConfig:
    openai_config: OpenAIEncoderConfig = field(default_factory=OpenAIEncoderConfig)
    ollama_config: OllamaEncoderConfig = field(default_factory=OllamaEncoderConfig)
    hf_config: HFEncoderConfig = field(default_factory=HFEncoderConfig)
    jina_config: JinaEncoderConfig = field(default_factory=JinaEncoderConfig)
    cohere_config: CohereEncoderConfig = field(default_factory=CohereEncoderConfig)


class TestEncode:
    cfg: EncodeTestConfig = OmegaConf.merge(
        OmegaConf.structured(EncodeTestConfig),
        OmegaConf.load(
            os.path.join(os.path.dirname(__file__), "configs", "encoder.yaml")
        ),
    )
    text = ["Who is Bruce Wayne?"] * 2

    @pytest.mark.asyncio
    async def test_openai(self):
        encoder = OpenAIEncoder(self.cfg.openai_config)
        r1 = encoder.encode(self.text)
        r2 = await encoder.async_encode(self.text)
        assert (r1 - r2).max() < 1e-5
        return

    @pytest.mark.asyncio
    async def test_ollama(self):
        encoder = OllamaEncoder(self.cfg.ollama_config)
        r1 = encoder.encode(self.text)
        r2 = await encoder.async_encode(self.text)
        assert (r1 - r2).max() < 1e-5
        return

    @pytest.mark.asyncio
    async def test_hf(self):
        encoder = HFEncoder(self.cfg.hf_config)
        r1 = encoder.encode(self.text)
        r2 = await encoder.async_encode(self.text)
        assert (r1 - r2).max() < 1e-5
        return

    @pytest.mark.asyncio
    async def test_jina(self):
        encoder = JinaEncoder(self.cfg.jina_config)
        r1 = encoder.encode(self.text)
        r2 = await encoder.async_encode(self.text)
        assert (r1 - r2).max() < 1e-5
        return

    @pytest.mark.asyncio
    async def test_cohere(self):
        encoder = CohereEncoder(self.cfg.cohere_config)
        r1 = encoder.encode(self.text)
        r2 = await encoder.async_encode(self.text)
        assert (r1 - r2).max() < 1e-5
        return
