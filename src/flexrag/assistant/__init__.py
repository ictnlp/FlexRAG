from .assistant import ASSISTANTS, PREDEFINED_PROMPTS, AssistantBase, SearchHistory
from .basic_assistant import BasicAssistant, BasicAssistantConfig
from .chatqa_assistant import ChatQAAssistant
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
    "SearchHistory",
    "PREDEFINED_PROMPTS",
    "BasicAssistant",
    "BasicAssistantConfig",
    "ModularAssistant",
    "ModularAssistantConfig",
    "ChatQAAssistant",
    "JinaDeepSearch",
    "JinaDeepSearchConfig",
    "PerplexityAssistant",
    "PerplexityAssistantConfig",
]
