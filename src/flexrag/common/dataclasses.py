from __future__ import annotations

import json
from dataclasses import field
from os import PathLike
from typing import Annotated, Any, MutableSequence, Optional, Sequence

from pydantic import AfterValidator
from rich.console import Console
from rich.markdown import Markdown

from .base64_utils import (
    base64_to_binary,
    base64_to_image,
    binary_to_base64,
    image_to_base64,
)
from .configure import Choices, data

console = Console()


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
    :type content: str | list[dict[str, Any]]

    For standard text-based messages, `content` is typically a string.

    For example:

        >>> turn = ChatTurn(role="user", content="Hello, how are you?")

    For rich content (e.g., images, files), `content` can be a list of
    dictionaries, each specifying a content type and its associated data.

    For example:

        >>> turn = ChatTurn(
        ...    role="assistant",
        ...    content=[
        ...        {"type": "text", "text": "Here is an image for you."},
        ...        {"type": "image", "url": "http://example.com/image.png"},
        ...    ]
        ... )

    Currently, supported content types in the dictionaries include:

        - Text: ``{"type": "text", "text": "<text content>"}``
        - Reasoning: ``{"type": "reasoning", "text": "<reasoning content>"}``
        - Image by URL: ``{"type": "image", "url": "<image url>"}``
        - Image by PIL Image: ``{"type": "image", "image": <PIL Image object>}``
        - Image by file path: ``{"type": "image", "image_path": "<path to image file>"}``
        - PDF by URL: ``{"type": "pdf", "url": "<pdf url>"}``
        - PDF by file path: ``{"type": "pdf", "file_path": "<path to pdf file>"}``
        - PDF by binary data: ``{"type": "pdf", "binary": <bytes or bytearray>}``
        - Audio by URL: ``{"type": "audio", "url": "<audio url>"}``
        - Audio by file path: ``{"type": "audio", "file_path": "<path to audio file>"}``
        - Audio by binary data: ``{"type": "audio", "binary": <bytes or bytearray>}``
        - Video by URL: ``{"type": "video", "url": "<video url>"}``
        - Video by file path: ``{"type": "video", "file_path": "<path to video file>"}``
        - Video by binary data: ``{"type": "video", "binary": <bytes or bytearray>}``
    """

    role: Annotated[str, Choices("user", "assistant", "system")]
    content: str | list[dict[str, Any]]

    def to_dict(self, pure_text: bool = False) -> dict[str, str | list[dict[str, Any]]]:
        """Converts the ChatTurn instance to a dictionary.

        :param pure_text: Whether to encode binary fields to base64. Default is False.
            If True, binary data (e.g. bytes, PIL Image) will be converted
            into base64 strings, but the overall structure of content
            (str or list[dict]) will be preserved so that it can be fully
            restored.
        :type pure_text: bool
        :return: A dictionary representation of the ChatTurn.
        :rtype: dict[str, str | list[dict[str, Any]]]
        """
        if not pure_text:
            return {"role": self.role, "content": self.content}

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        encoded_content: list[dict[str, Any]] = []
        for part in self.content:
            ctype = part.get("type")
            new_part: dict[str, Any] = dict(part)
            if ctype == "image":
                if "image" in new_part:
                    new_part["image"] = image_to_base64(new_part["image"])
                    new_part["encoding"] = "base64"
            elif ctype in {"pdf", "audio", "video"}:
                if "binary" in new_part:
                    new_part["binary"] = binary_to_base64(new_part["binary"])
                    new_part["encoding"] = "base64"
            encoded_content.append(new_part)
        return {"role": self.role, "content": encoded_content}

    @classmethod
    def from_dict(cls, chat_turn: dict[str, str | list[dict[str, Any]]]) -> ChatTurn:
        role = chat_turn.get("role")
        content = chat_turn.get("content")
        if role is None or content is None:
            raise ValueError("chat_turn must have 'role' and 'content' fields")
        if isinstance(content, str):
            return cls(role=role, content=content)
        if not isinstance(content, list):
            raise ValueError("content must be either str or list[dict]")

        restored_content: list[dict[str, Any]] = []
        for part in content:
            ctype = part.get("type")
            encoding = part.get("encoding")
            new_part: dict[str, Any] = dict(part)
            if encoding == "base64":
                if ctype == "image" and "image" in part:
                    new_part["image"] = base64_to_image(part["image"])
                elif ctype in {"pdf", "audio", "video"} and ("binary" in part):
                    new_part["binary"] = base64_to_binary(part["binary"])
                new_part.pop("encoding")
            restored_content.append(new_part)

        return cls(role=role, content=restored_content)

    def pretty_print(self) -> None:
        header = f"[bold cyan]{self.role.upper()}[/bold cyan]"
        console.print(header)
        if isinstance(self.content, str):
            console.print(Markdown(self.content))
        else:
            new_text = ""
            for content_part in self.content:
                if content_part.get("type") == "text":
                    new_text += content_part.get("text", "")
                else:
                    new_text += f"\n[{content_part.get('type', 'unknown')} content]\n"
            console.print(Markdown(new_text))
        return

    @property
    def text_content(self) -> str | None:
        """Returns the text content of the chat turn, if any."""
        if isinstance(self.content, str):
            return self.content
        texts = []
        for part in self.content:
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "".join(texts) if texts else None


def validate_chat_messages(chat_messages: list[ChatTurn]) -> list[ChatTurn]:
    """Validates the chat messages.

    Steps:

    1. Check that the input is a list.
    2. Check that every item is a :class:`ChatTurn`.
    3. Check that roles follow the required alternation rules.

    :param chat_messages: The chat messages to validate.
    :type chat_messages: list[ChatTurn]
    :return: A list of ChatTurn instances after validation.
    :rtype: list[ChatTurn]
    """
    # step 1: check that the input is a list
    if not isinstance(chat_messages, list):
        raise TypeError("chat_messages must be a list[ChatTurn]")

    # Step2: check that every item is a ChatTurn
    for n, turn in enumerate(chat_messages):
        if not isinstance(turn, ChatTurn):
            raise TypeError(f"The item at index {n} is not a ChatTurn instance")

    # step 3: check role alternation
    if len(chat_messages) == 0:
        return chat_messages
    if chat_messages[0].role == "system":
        even_role = "assistant"
        odd_role = "user"
    elif chat_messages[0].role == "user":
        even_role = "user"
        odd_role = "assistant"
    else:
        raise ValueError(
            f"The role of the first chat turn must be 'system' or 'user', "
            f"but got '{chat_messages[0].role}'"
        )
    if len(chat_messages) == 1:
        return chat_messages
    for n, turn in enumerate(chat_messages[1:], start=1):
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
    return chat_messages


@data
class ChatMessages(MutableSequence[ChatTurn]):
    """
    ChatMessages represents the full message history in a single chat session.
    Internally it stores an ordered list of `ChatTurn` objects and enforces a
    valid alternation of roles (system / user / assistant).

    This class implements `MutableSequence[ChatTurn]`, so you can work with it
    much like a regular Python list (indexing, appending, inserting, deleting),
    while also benefiting from additional helpers for validation and
    serialization:

    - `to_list()` / `from_list()`:
      Convert between `ChatMessages` and `list[dict]` (each dict containing
      "role" and "content"), which is often the format used by LLM SDKs
      (OpenAI, Azure, local models, etc.).
    - `to_json(path)` / `from_json(path)`:
      Save the conversation history to a JSON file or restore it from one, for
      logging, reproducibility, or annotation workflows.
    - `system` / `set_system(content)`:
      Get or set the system prompt. `set_system` ensures that the system
      message is always the first turn in the conversation.
    - `copy()`:
      Create a shallow copy of the current history so you can modify it
      (e.g., inject extra turns, rewrite prompts) without mutating the
      original conversation.

    Role convention and validation rules:

    1. A conversation may optionally start with a system message
       (`role="system"`). After that, messages must start with a user turn and
       strictly alternate between user and assistant:
       system -> user -> assistant -> user -> assistant -> ...
    2. If the first message is system, then indices 1, 3, 5, ... must be user
       and indices 2, 4, 6, ... must be assistant. If the first message is
       user, then indices 1, 3, 5, ... must be assistant and indices 2, 4, 6,
       ... must be user.
    3. The validation function `validate_chat_messages` is applied via
       `AfterValidator`, so constructing, inserting, or assigning items that
       would break the role alternation will raise `ValueError`.

    Typical usage example:

    .. code-block:: python

        # Build a conversation from scratch
        msgs = ChatMessages()
        msgs.set_system("You are a helpful assistant.")
        msgs.append(ChatTurn(role="user", content="Hi, please write a sort function."))
        msgs.append(ChatTurn(role="assistant", content="Sure, here is a Python example..."))

        # Convert to list / JSON for model calls or persistence
        payload = msgs.to_list()
        msgs.to_json("conversation.json")

        # Restore from an existing conversation
        loaded = ChatMessages.from_json("conversation.json")
        print(loaded.system)

    ChatMessages is intended to be the unified, type-safe container for dialog
    history when building RAG pipelines, agents, or multi-turn QA systems. It
    helps keep role sequences valid and makes it easy to move between in-memory
    objects, JSON files, and model-specific input formats.
    """

    history: Annotated[
        list[ChatTurn],
        AfterValidator(validate_chat_messages),
    ] = field(default_factory=list)

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

    def to_list(self, pure_text: bool = False) -> list[dict[str, str]]:
        """Converts the chat messages to a list of dictionaries."""
        return [turn.to_dict(pure_text) for turn in self.history]

    def to_json(self, path: str | PathLike):
        """Saves the chat messages to a JSON file.

        :param path: The path to save the JSON file.
        :type path: str | PathLike
        """
        data = [turn.to_dict(pure_text=True) for turn in self.history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    @classmethod
    def from_list(cls, messages: Sequence[ChatTurn | dict[str, Any]]) -> ChatMessages:
        """Creates a ChatMessages instance from a sequence of dictionaries.
        :param messages: A sequence of dictionaries representing chat turns.
        :type messages: Sequence[ChatTurn | dict[str, Any]]
        :return: An instance of ChatMessages.
        :rtype: ChatMessages
        """
        turns: list[ChatTurn] = []
        for turn in messages:
            if isinstance(turn, dict):
                turns.append(ChatTurn.from_dict(turn))
            else:
                turns.append(turn)
        return cls(history=turns)

    @classmethod
    def from_json(cls, path: str | PathLike) -> ChatMessages:
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

    @system.setter
    def system(self, content: str | None) -> None:
        """Sets the system prompt.

        :param content: The content of the system prompt.
            If None, removes the system prompt if it exists.
        :type content: str | None
        """
        if content is None:
            if len(self.history) > 0 and self.history[0].role == "system":
                self.history.pop(0)
            return
        if len(self.history) > 0 and self.history[0].role == "system":
            self.history[0].content = content
        else:
            self.history.insert(0, ChatTurn(role="system", content=content))
        return

    def copy(self) -> ChatMessages:
        """Creates a copy of the ChatMessages instance."""
        return ChatMessages(history=self.history.copy())

    def pretty_print(self) -> None:
        turn_num = 0
        for n, turn in enumerate(self.history):
            if n == 0 and turn.role == "system":
                console.print(
                    f"\n[bold yellow]--- System Prompt ---[/bold yellow]\n",
                    justify="center",
                )
                turn.pretty_print()
                continue
            if turn.role == "user":
                turn_num += 1
                console.print(
                    f"\n[bold yellow]--- Turn {turn_num} ---[/bold yellow]\n",
                    justify="center",
                )
            turn.pretty_print()
        return
