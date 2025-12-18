from typing import Optional

from flexrag.common import Register, data
from flexrag.common.dataclasses import ChatMessages, Context, RetrievedContext


@data
class AssistantResponse:
    """The response from the assistant.

    :param response: The response to the question.
    :type response: str
    :param contexts: The contexts used to answer the question.
        Defaults to None.
    :type contexts: Optional[list[RetrievedContext]], optional
    :param metadata: The metadata of the assistant.
        Defaults to None.
    :type metadata: Optional[dict], optional
    """

    response: str
    contexts: Optional[list[RetrievedContext]] = None
    metadata: Optional[dict] = None


class AssistantBase:
    """AssistantBase defines the interface for interactions between RAG pipelines and RAG tasks.
    While it closely resembles traditional chat models in structure,
    its return values differ by including not only the response
    but also retrieved passages and metadata that can be leveraged for further analysis.

    The subclasses of this base class should implement at least one of the methods
    `answer` or `answer_with_contexts`.
    The `answer` method is used to generate a response to the given messages,
    while the `answer_with_contexts` method is used to generate a response
    to the given messages with the provided contexts.
    """

    def answer(
        self,
        messages: ChatMessages | list[dict],
        disable_retrieval: bool = False,
    ) -> AssistantResponse:
        """Generate a response to the given messages.

        :param messages: The messages to generate a response for.
        :type messages: ChatMessages | list[dict]
        :param disable_retrieval: If True, disables retrieval and uses the messages directly.
            Defaults to False.
        :type disable_retrieval: bool, optional
        :return: A dataclass containing the following elements:
            * The response to the question.
            * The contexts used to answer the question.
            * The metadata of the assistant.
        :rtype: AssistantResponse
        """
        raise NotImplementedError(
            "The `answer` method is not implemented. "
            "Please implement the `answer` method in the subclass."
        )

    def answer_with_contexts(
        self,
        messages: ChatMessages | list[dict],
        contexts: list[Context],
    ) -> AssistantResponse:
        """Generate a response to the given messages with the provided contexts.

        :param messages: The messages to generate a response for.
        :type messages: ChatMessages | list[dict]
        :param contexts: The contexts to incorporate into the response.
        :type contexts: list[Context]
        :return: A dataclass containing the following elements:
            * The response to the question.
            * The contexts used to answer the question.
            * The metadata of the assistant.
        :rtype: AssistantResponse
        """
        raise NotImplementedError(
            "The `answer_with_contexts` method is not implemented. "
            "Please implement the `answer_with_contexts` method in the subclass."
        )


ASSISTANTS = Register[AssistantBase]("assistant")


# PREDEFINED_PROMPTS = {
#     "shortform_with_context": ChatMessages.from_json(
#         os.path.join(
#             os.path.dirname(__file__),
#             "assistant_prompts",
#             "shortform_generate_prompt_with_context.json",
#         )
#     ),
#     "shortform_without_context": ChatMessages.from_json(
#         os.path.join(
#             os.path.dirname(__file__),
#             "assistant_prompts",
#             "shortform_generate_prompt_without_context.json",
#         )
#     ),
#     "longform_with_context": ChatMessages.from_json(
#         os.path.join(
#             os.path.dirname(__file__),
#             "assistant_prompts",
#             "longform_generate_prompt_with_context.json",
#         )
#     ),
#     "longform_without_context": ChatMessages.from_json(
#         os.path.join(
#             os.path.dirname(__file__),
#             "assistant_prompts",
#             "longform_generate_prompt_without_context.json",
#         )
#     ),
# }
