from abc import ABC, abstractmethod
from typing import Optional

from flexrag.common import Register, data
from flexrag.common.dataclasses import ChatMessages, ChatTurn, Context, RetrievedContext


@data
class AssistantResponse:
    """The response from the assistant.

    :param response: The response to the question.
    :type response: ChatTurn
    :param contexts: The contexts used to answer the question.
        Defaults to None.
    :type contexts: Optional[list[RetrievedContext]], optional
    :param metadata: The metadata of the assistant.
        Defaults to None.
    :type metadata: Optional[dict], optional
    """

    response: ChatTurn
    contexts: Optional[list[RetrievedContext]] = None
    metadata: Optional[dict] = None


class AssistantBase(ABC):
    """AssistantBase defines the interface for interactions between a LLM Agent
    and FlexRAG tasks. While it closely resembles traditional chat models in
    structure, its return values differ by including not only the response but
    also retrieved passages and metadata that can be leveraged for further
    analysis.

    The subclasses of this base class should implement the methods `answer` to
    generate response based on the given messages and contexts.
    """

    @abstractmethod
    def answer(
        self,
        messages: ChatMessages | list[dict],
        additional_sessions: list[ChatMessages] | None = None,
    ) -> AssistantResponse:
        """Generate a response to the given messages.

        :param messages: The messages to generate a response for.
        :type messages: ChatMessages | list[dict]
        :param additional_sessions: The additional conversation sessions that
            may be relevant to the current conversation. Defaults to None.
        :type additional_sessions: Optional[list[ChatMessages]], optional
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


ASSISTANTS = Register[AssistantBase]("assistant")
