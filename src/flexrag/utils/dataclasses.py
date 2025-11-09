import json
from dataclasses import field
from os import PathLike
from typing import Annotated, Any, MutableSequence, Optional

from pydantic import AfterValidator

from .configure import Choices, data


@data
class Context:
    """The dataclass for retrieved context.

    :param context_id: The unique identifier of the context. Default: None.
    :type context_id: Optional[str]
    :param data: The context data. Default: {}.
    :type data: dict
    :param source: The source of the retrieved data. Default: None.
    :type source: Optional[str]
    :param meta_data: The metadata of the context. Default: {}.
    :type meta_data: dict
    """

    context_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    source: Optional[str] = None
    meta_data: dict = field(default_factory=dict)


@data
class RetrievedContext(Context):
    """The dataclass for retrieved context.

    :param retriever: The name of the retriever. Required.
    :type retriever: str
    :param query: The query for retrieval. Required.
    :type query: str
    :param score: The relevance score of the retrieved data. Default: 0.0.
    :type score: float
    """

    retriever: str = ""
    query: str = ""
    score: float = 0.0


@data
class ChatTurn:
    """ChatTurn is a dataclass that represents a single turn in a chat session.

    :param role: The role of the chat turn, can be "user", "assistant", or "system".
    :type role: str
    :param content: The content of the chat turn, can be a string or a list of dictionaries.
    :type content: str | dict[str, Any]
    """

    role: Annotated[str, Choices("user", "assistant", "system")]
    content: str | list[dict[str, Any]]

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, chat_turn: dict[str, str]):
        return cls(role=chat_turn["role"], content=chat_turn["content"])


def validate_chat_messages(
    chat_messages: list[ChatTurn] | list[dict[str, str]],
) -> list[ChatTurn]:
    """Validates the chat messages.

    :param chat_messages: The chat messages to validate.
    :type chat_messages: list[ChatTurn] | list[dict[str, str]]
    :return: A list of ChatTurn instances.
    :rtype: list[ChatTurn]
    """
    if not isinstance(chat_messages, list):
        raise ValueError("chat_messages must be a list")
    messages = [
        ChatTurn.from_dict(turn) if isinstance(turn, dict) else turn
        for turn in chat_messages
    ]
    if len(messages) == 0:
        return messages
    if messages[0].role == "system":
        even_role = "assistant"
        odd_role = "user"
    elif messages[0].role == "user":
        even_role = "user"
        odd_role = "assistant"
    else:
        raise ValueError(
            f"The role of the first chat turn must be 'system' or 'user', "
            f"but got '{messages[0].role}'"
        )
    if len(messages) == 1:
        return messages
    for n, turn in enumerate(messages[1:], start=1):
        if n % 2 == 0:
            if turn.role != even_role:
                raise ValueError(
                    f"The role of the chat turn at index {n} must be '{even_role}', "
                    f"but got '{turn.role}'"
                )
        else:
            if turn.role != odd_role:
                raise ValueError(
                    f"The role of the chat turn at index {n} must be '{odd_role}', "
                    f"but got '{turn.role}'"
                )
    return messages


@data
class ChatMessages(MutableSequence[ChatTurn]):
    """
    ChatMessages is a dataclass that represents the chat messages in a chat session.
    """

    history: Annotated[list[ChatTurn], AfterValidator(validate_chat_messages)] = field(
        default_factory=list
    )

    def __getitem__(self, index: int) -> ChatTurn:
        """Returns the chat turn at the specified index."""
        return self.history[index]

    def __setitem__(self, index: int, chat_turn: ChatTurn | dict[str, str]) -> None:
        """Sets the chat turn at the specified index."""
        if isinstance(chat_turn, dict):
            chat_turn = ChatTurn.from_dict(chat_turn)
        self.history[index] = chat_turn
        return

    def __delitem__(self, index: int) -> None:
        """Deletes the chat turn at the specified index."""
        del self.history[index]
        return

    def insert(self, index: int, chat_turn: ChatTurn | dict[str, str]) -> None:
        """Inserts a chat turn at the specified index."""
        if isinstance(chat_turn, dict):
            chat_turn = ChatTurn.from_dict(chat_turn)
        self.history.insert(index, chat_turn)
        return

    def __len__(self) -> int:
        """Returns the total number of chat turns in the prompt"""
        return len(self.history)

    def to_list(self) -> list[dict[str, str]]:
        """Converts the chat messages to a list of dictionaries."""
        return [turn.to_dict() for turn in self.history]

    def to_json(self, path: str | PathLike):
        """Saves the chat messages to a JSON file.

        :param path: The path to save the JSON file.
        :type path: str | PathLike
        """
        data = [turn.to_dict() for turn in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    @classmethod
    def from_list(cls, prompt: list[dict[str, str]]) -> "ChatMessages":
        """Creates a ChatMessages instance from a list of dictionaries.
        :param prompt: A list of dictionaries representing chat turns.
        :type prompt: list[dict[str, str]]
        :return: An instance of ChatMessages.
        :rtype: ChatMessages
        """
        turns = [ChatTurn.from_dict(turn) for turn in prompt]
        return cls(history=turns)

    @classmethod
    def from_json(cls, path: str | PathLike) -> "ChatMessages":
        """Loads the chat messages from a JSON file.

        :param path: The path to the JSON file.
        :type path: str | PathLike
        :return: An instance of ChatMessages.
        :rtype: ChatMessages
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_list(data)

    @property
    def system(self) -> Optional[str]:
        """Returns the system message if it exists, otherwise None."""
        if len(self.history) > 0 and self.history[0].role == "system":
            return self.history[0].content
        return None

    def set_system(self, content: str) -> None:
        """Sets the system prompt.

        :param content: The content of the system prompt.
        :type content: str
        """
        if len(self.history) > 0 and self.history[0].role == "system":
            self.history[0].content = content
        else:
            self.history.insert(0, ChatTurn(role="system", content=content))
        return

    def copy(self) -> "ChatMessages":
        """Creates a copy of the ChatMessages instance."""
        return ChatMessages(history=self.history.copy())
