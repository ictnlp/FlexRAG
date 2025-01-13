from .anthropic_model import AnthropicGenerator, AnthropicGeneratorConfig
from .cohere_model import (
    CohereEncoder,
    CohereEncoderConfig,
)
from .hf_model import (
    HFModelConfig,
    HFEncoder,
    HFEncoderConfig,
    HFGenerator,
    HFGeneratorConfig,
    HFClipEncoder,
    HFClipEncoderConfig,
    HFVLMGenerator,
    HFVLMGeneratorConfig,
)
from .jina_model import JinaEncoder, JinaEncoderConfig
from .llamacpp_model import LlamacppGenerator, LlamacppGeneratorConfig
from .model_base import (
    EncoderBase,
    GenerationConfig,
    GeneratorBase,
    VLMGeneratorBase,
    GENERATORS,
    ENCODERS,
)
from .ollama_model import (
    OllamaGenerator,
    OllamaGeneratorConfig,
    OllamaEncoder,
    OllamaEncoderConfig,
)
from .openai_model import (
    OpenAIConfig,
    OpenAIEncoder,
    OpenAIEncoderConfig,
    OpenAIGenerator,
    OpenAIGeneratorConfig,
)
from .vllm_model import VLLMGenerator, VLLMGeneratorConfig
from .sentence_transformers_model import (
    SentenceTransformerEncoder,
    SentenceTransformerEncoderConfig,
)


GeneratorConfig = GENERATORS.make_config(config_name="GeneratorConfig")
EncoderConfig = ENCODERS.make_config(config_name="EncoderConfig", default=None)


__all__ = [
    "GeneratorBase",
    "VLMGeneratorBase",
    "GenerationConfig",
    "EncoderBase",
    "AnthropicGenerator",
    "AnthropicGeneratorConfig",
    "HFModelConfig",
    "HFGenerator",
    "HFGeneratorConfig",
    "HFEncoder",
    "HFEncoderConfig",
    "HFClipEncoder",
    "HFClipEncoderConfig",
    "HFVLMGenerator",
    "HFVLMGeneratorConfig",
    "OllamaGenerator",
    "OllamaGeneratorConfig",
    "OllamaEncoder",
    "OllamaEncoderConfig",
    "OpenAIGenerator",
    "OpenAIGeneratorConfig",
    "OpenAIConfig",
    "OpenAIEncoder",
    "OpenAIEncoderConfig",
    "VLLMGenerator",
    "VLLMGeneratorConfig",
    "LlamacppGenerator",
    "LlamacppGeneratorConfig",
    "JinaEncoder",
    "JinaEncoderConfig",
    "CohereEncoder",
    "CohereEncoderConfig",
    "SentenceTransformerEncoder",
    "SentenceTransformerEncoderConfig",
    "GENERATORS",
    "ENCODERS",
]
