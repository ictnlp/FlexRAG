from copy import deepcopy
from dataclasses import dataclass, field

from kylin.kylin_prompts import generate_prompt
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
from kylin.utils import Choices


@dataclass
class AssistantConfig:
    generator_type: Choices(["vllm", "openai", "hf", "ollama"]) = "openai"  # type: ignore
    openai_config: OpenAIGeneratorConfig = field(default_factory=OpenAIGeneratorConfig)
    vllm_config: VLLMGeneratorConfig = field(default_factory=VLLMGeneratorConfig)
    hf_config: HFGeneratorConfig = field(default_factory=HFGeneratorConfig)
    ollama_config: OllamaGeneratorConfig = field(default_factory=OllamaGeneratorConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    adaptive_retrieval: bool = False


class Assistant:
    def __init__(self, cfg: AssistantConfig):
        self.adaptive_retrieval = cfg.adaptive_retrieval
        self.gen_cfg = cfg.generation_config
        self.generator = self.load_generator(
            cfg.generator_type,
            vllm_cfg=cfg.vllm_config,
            hf_cfg=cfg.hf_config,
            openai_cfg=cfg.openai_config,
            ollama_cfg=cfg.ollama_config,
        )
        return

    def load_generator(
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

    def answer(self, question: str, contexts: list) -> tuple[str, list[dict[str, str]]]:
        """Answer question with given contexts

        Args:
            question (str): The question to answer.
            contexts (list): The contexts searched by the searcher.

        Returns:
            response (str): response to the question
            prompt (list[dict[str, str]]): prompt used.
        """
        if self.adaptive_retrieval:
            if not self.decide_search(question):
                contexts = []

        if len(contexts) > 0:
            prompt = deepcopy(generate_prompt["with_contexts"])
        else:
            prompt = deepcopy(generate_prompt["without_contexts"])
        usr_prompt = ""
        for n, context in enumerate(contexts):
            if "summary" in context:
                ctx = context["summary"]
            else:
                ctx = context["text"]
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"

        prompt.append({"role": "user", "content": usr_prompt})
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)[0]
        return response, prompt

    def decide_search(self, question: str) -> bool:
        raise NotImplementedError
