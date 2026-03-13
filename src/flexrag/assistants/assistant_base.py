from abc import ABC, abstractmethod
from collections.abc import Iterable
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
    def answer(self, messages: ChatMessages | list[dict]) -> AssistantResponse:
        """Generate a response to the given messages.

        :param messages: The messages to generate a response for.
        :type messages: ChatMessages | list[dict]
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

    def add_histories(self, histories: list[ChatMessages]) -> None:
        """Add conversation histories to the assistant. This method can be
        overridden by subclasses to maintain conversation state across multiple
        interactions. This interface is usually used for evaluating the memory
        ability of the assistant.

        :param histories: A list of conversation histories to add.
        :type histories: list[ChatMessages]
        """
        return

    def clear_histories(self) -> None:
        """Clear the conversation histories maintained by the assistant. This
        method can be overridden by subclasses to reset conversation state.
        This interface is usually used for evaluating the memory ability of the
        assistant.
        """
        return

    def add_contexts(self, contexts: Iterable[Context]) -> None:
        """Add contexts to the assistant. This method can be overridden by subclasses
        to maintain context state across multiple interactions. This interface is
        usually used for evaluating the RAG ability of the assistant.

        :param contexts: An iterable of contexts to add.
        :type contexts: Iterable[Context]
        """
        return

    def clear_contexts(self) -> None:
        """Clear the contexts maintained by the assistant. This method can be overridden
        by subclasses to reset context state. This interface is usually used for
        evaluating the RAG ability of the assistant.
        """
        return


ASSISTANTS = Register[AssistantBase]("assistant")
