from flexrag.common import LOGGER_MANAGER
from flexrag.common.dataclasses import ChatMessages, RetrievedContext
from flexrag.models.generators import GeneratorProtocol
from flexrag.processors.rankers.ranker_base import RankerBase
from flexrag.processors.refiners.refiner_base import RefinerProtocol
from flexrag.retrievers import FlexRetriever

from .assistant_base import ASSISTANTS, AssistantResponse
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

    def __init__(
        self,
        cfg: ModularAssistantConfig,
        generator: GeneratorProtocol,
        retriever: FlexRetriever | None = None,
        reranker: RankerBase | None = None,
        refiners: list[RefinerProtocol] | None = None,
    ):
        super().__init__(
            cfg,
            generator=generator,
            retriever=retriever,
            reranker=reranker,
            refiners=refiners,
        )
        logger.warning(
            f"ChatQA Assistant expects the model to be one of {self.allowed_models}."
        )
        return

    def answer_with_contexts(
        self, messages: ChatMessages, contexts: list[RetrievedContext] = []
    ) -> AssistantResponse:
        prefix = self.get_formatted_input(messages.copy(), contexts)
        response = self.generator.generate([prefix], generation_config=self.gen_cfg)
        return AssistantResponse(
            response=response[0][0], contexts=contexts, metadata={"prefix": prefix}
        )

    def get_formatted_input(
        self, messages: ChatMessages, contexts: list[RetrievedContext]
    ) -> str:
        # prepare system prompts
        prefix = f"{self.sys_prompt}\n\n"

        # add instruction to the first user message
        for item in messages:
            if item.role == "user":
                item.content = self.instruction + " " + item.content
                break

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

        # format prefix
        conversation = (
            "\n\n".join(
                [
                    (
                        "User: " + item.content
                        if item.role == "user"
                        else "Assistant: " + item.content
                    )
                    for item in messages
                ]
            )
            + "\n\nAssistant:"
        )
        prefix = f"{self.sys_prompt}\n\n{ctx}\n\n{conversation}"
        return prefix
