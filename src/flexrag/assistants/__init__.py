from .assistant_base import ASSISTANTS, AssistantBase, AssistantResponse
from .modular_rag_assistant import ModularAssistant, ModularAssistantConfig
from .online_assistant import (
    JinaDeepSearch,
    JinaDeepSearchConfig,
    PerplexityAssistant,
    PerplexityAssistantConfig,
)

AssistantConfig = ASSISTANTS.make_config(config_name="AssistantConfig")


__all__ = [
    "ASSISTANTS",
    "AssistantBase",
    "AssistantResponse",
    "ModularAssistant",
    "ModularAssistantConfig",
    "JinaDeepSearch",
    "JinaDeepSearchConfig",
    "PerplexityAssistant",
    "PerplexityAssistantConfig",
]
