from flexrag.common_dataclass import RetrievedContext
from flexrag.utils import LOGGER_MANAGER

from .assistant import ASSISTANTS
from .modular_rag_assistant import ModularAssistant, ModularAssistantConfig


logger = LOGGER_MANAGER.get_logger("flexrag.assistant.chatqa")


@ASSISTANTS("chatqa", config_class=ModularAssistantConfig)
class ChatQAAssistant(ModularAssistant):
    """The Modular assistant that employs the ChatQA model for response generation."""

    sys_prompt = (
        "System: This is a chat between a user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. "
        "The assistant should also indicate when the answer cannot be found in the context."
    )
    instruction = "Please give a full and complete answer for the question."
    allowed_models = [
        "nvidia/Llama3-ChatQA-2-8B",
        "nvidia/Llama3-ChatQA-2-70B",
        "nvidia/Llama3-ChatQA-1.5-8B",
        "nvidia/Llama3-ChatQA-1.5-70B",
    ]

    def __init__(self, cfg: ModularAssistantConfig):
        super().__init__(cfg)
        logger.warning(
            f"ChatQA Assistant expects the model to be one of {self.allowed_models}."
        )
        return

    def answer_with_contexts(
        self, question: str, contexts: list[RetrievedContext]
    ) -> tuple[str, str]:
        prefix = self.get_formatted_input(question, contexts)
        response = self.generator.generate([prefix], generation_config=self.gen_cfg)
        return response[0][0], prefix

    def get_formatted_input(
        self, question: str, contexts: list[RetrievedContext]
    ) -> str:
        # prepare system prompts
        prefix = f"{self.sys_prompt}\n\n"

        # prepare context string
        for n, context in enumerate(contexts):
            if len(self.used_fields) == 0:
                ctx = ""
                for field_name, field_value in context.data.items():
                    ctx += f"{field_name}: {field_value}\n"
            elif len(self.used_fields) == 1:
                ctx = context.data[self.used_fields[0]]
            else:
                ctx = ""
                for field_name in self.used_fields:
                    ctx += f"{field_name}: {context.data[field_name]}\n"
            prefix += f"Context {n + 1}: {ctx}\n\n"

        # prepare user instruction
        prefix += f"User: {self.instruction} {question}\n\nAssistant:"
        return prefix
