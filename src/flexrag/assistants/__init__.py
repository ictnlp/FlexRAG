from .assistant_base import (
    ASSISTANTS,
    AssistantBase,
    AssistantProtocol,
    AssistantResult,
)
from .modular_assistant import ModularAssistant, ModularAssistantConfig
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
    "AssistantProtocol",
    "AssistantResult",
    "ModularAssistant",
    "ModularAssistantConfig",
    "JinaDeepSearch",
    "JinaDeepSearchConfig",
    "PerplexityAssistant",
    "PerplexityAssistantConfig",
]
