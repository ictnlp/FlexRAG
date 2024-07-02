import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from kylin.kylin_prompts import *
from kylin.models import (
    GenerationConfig,
    GeneratorBase,
    HFGenerator,
    HFGeneratorConfig,
    OllamaGenerator,
    OllamaGeneratorConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
    VLLMGenerator,
    VLLMGeneratorConfig,
)
from kylin.retriever import Retriever


logger = logging.getLogger(__name__)


# fmt: off
@dataclass
class SearcherConfig:
    agent_type: str = "openai"
    vllm_config: Optional[VLLMGeneratorConfig] = field(default_factory=VLLMGeneratorConfig)
    hf_config: Optional[HFGeneratorConfig] = field(default_factory=HFGeneratorConfig)
    openai_config: Optional[OpenAIGeneratorConfig] = field(default_factory=OpenAIGeneratorConfig)
    ollama_config: Optional[OllamaGeneratorConfig] = field(default_factory=OllamaGeneratorConfig)
    agent_gen_cfg: GenerationConfig = field(default_factory=GenerationConfig)
# fmt: on


class BaseSearcher(ABC):
    retriever: Retriever
    def __init__(self, cfg: SearcherConfig) -> None:
        self.agent = self.load_agent(
            cfg.agent_type,
            vllm_cfg=cfg.vllm_config,
            hf_cfg=cfg.hf_config,
            openai_cfg=cfg.openai_config,
            ollama_cfg=cfg.ollama_config,
        )
        self.gen_cfg = cfg.agent_gen_cfg
        return

    @abstractmethod
    def search(
        self, question: str
    ) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
        return

    def load_agent(
        self,
        model_type: str,
        vllm_cfg: VLLMGeneratorConfig,
        hf_cfg: HFGeneratorConfig,
        openai_cfg: OpenAIGeneratorConfig,
        ollama_cfg: OllamaGeneratorConfig,
    ) -> GeneratorBase:
        match model_type:
            case "vllm":
                model = VLLMGenerator(vllm_cfg)
            case "hf":
                model = HFGenerator(hf_cfg)
            case "openai":
                model = OpenAIGenerator(openai_cfg)
            case "ollama":
                model = OllamaGenerator(ollama_cfg)
            case _:
                raise ValueError(f"Not supported model: {model_type}")
        return model

    def close(self) -> None:
        self.retriever.close()
        return
