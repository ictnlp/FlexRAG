from __future__ import annotations

import json
from dataclasses import field
from os import PathLike
from typing import (
    Annotated,
    Any,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    TypeAlias,
    TypedDict,
    cast,
)

from PIL.ImageFile import ImageFile
from pydantic import AfterValidator, Field, ValidationInfo
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


class TextContentPart(TypedDict):
    type: Literal["text"]
    text: str


class ToolCallContentPart(TypedDict):
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict[str, Any] | str


class ImageContentPart(TypedDict, total=False):
    type: Literal["image"]
    url: str
    image: ImageFile
    image_path: str


class PDFContentPart(TypedDict, total=False):
    type: Literal["pdf"]
    url: str
    file_path: str
    binary: bytes | bytearray


class FileContentPart(TypedDict, total=False):
    type: Literal["file"]
    url: str
    file_path: str
    binary: bytes | bytearray
    mime_type: str
    file_name: str


class AudioContentPart(TypedDict, total=False):
    type: Literal["audio"]
    url: str
    file_path: str
    binary: bytes | bytearray


class VideoContentPart(TypedDict, total=False):
    type: Literal["video"]
    url: str
    file_path: str
    binary: bytes | bytearray


ContentPart: TypeAlias = (
    TextContentPart
    | ToolCallContentPart
    | ImageContentPart
    | PDFContentPart
    | FileContentPart
    | AudioContentPart
    | VideoContentPart
)


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

    :param retriever: The name of the retriever. Default: None.
    :type retriever: Optional[str]
    :param query: The query for retrieval. Default: None.
    :type query: Optional[str]
    :param score: The relevance score of the retrieved data. Default: None.
    :type score: Optional[float]
    """

    retriever: Optional[str] = None
    query: Optional[str] = None
    score: Optional[float] = None


def _valid_chat_role(role: str, info: ValidationInfo) -> str:
    """Validates the chat role.

    :param role: The role to validate.
    :type role: str
    :return: The validated role.
    :rtype: str
    """
    strict_mode = info.data.get("strict_mode", True)
    valid_roles = {"user", "assistant", "system", "tool"}
    if role not in valid_roles and strict_mode:
        raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}.")
    return role


@data(kw_only=True)
class ChatTurn:
    """ChatTurn is a dataclass that represents a single turn in a chat session.

    :param role: The role of the chat turn, can be "user", "assistant", "system",
        or "tool".
    :type role: str
    :param content: The content of the chat turn, can be a string or a list of
        typed content blocks.
    :type content: str | list[ContentPart]
    :param turn_id: The unique identifier for the chat turn. Default: None.
    :type turn_id: Optional[str]
    :param reasoning_content: Provider-normalized reasoning text for this turn.
        Default: None.
    :type reasoning_content: Optional[str]
    :param thinking_blocks: Provider-native structured thinking blocks for this
        turn. Default: None.
    :type thinking_blocks: Optional[list[dict[str, Any]]]
    :param tool_call_id: The tool call ID associated with this turn.
        Primarily used when ``role="tool"``. Default: None.
    :type tool_call_id: Optional[str]
    :param name: The tool name associated with this turn.
        Primarily used when ``role="tool"``. Default: None.
    :type name: Optional[str]
    :param metadata: Additional provider-specific metadata for the turn.
        Default: {}.
    :type metadata: dict[str, Any]
    :param strict_mode: Whether to enforce strict role validation. Default: True.
        If True, only "user", "assistant", "system", and "tool" are allowed
        as roles.
    :type strict_mode: bool

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
        - Tool call: ``{"type": "tool_call", "id": "<call id>", "name": "<tool name>", "arguments": <dict or str>}``
        - Image by URL: ``{"type": "image", "url": "<image url>"}``
        - Image by PIL Image: ``{"type": "image", "image": <PIL Image object>}``
        - Image by file path: ``{"type": "image", "image_path": "<path to image file>"}``
        - PDF by URL: ``{"type": "pdf", "url": "<pdf url>"}``
        - PDF by file path: ``{"type": "pdf", "file_path": "<path to pdf file>"}``
        - PDF by binary data: ``{"type": "pdf", "binary": <bytes or bytearray>}``
        - Generic file: ``{"type": "file", "file_path": "<path>", "mime_type": "<mime type>"}``
        - Audio by URL: ``{"type": "audio", "url": "<audio url>"}``
        - Audio by file path: ``{"type": "audio", "file_path": "<path to audio file>"}``
        - Audio by binary data: ``{"type": "audio", "binary": <bytes or bytearray>}``
        - Video by URL: ``{"type": "video", "url": "<video url>"}``
        - Video by file path: ``{"type": "video", "file_path": "<path to video file>"}``
        - Video by binary data: ``{"type": "video", "binary": <bytes or bytearray>}``
    """

    strict_mode: bool = field(default=True, repr=False)
    role: Annotated[str, AfterValidator(_valid_chat_role)]
    content: str | list[ContentPart]
    turn_id: Optional[str] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[list[dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, pure_text: bool = False) -> dict[str, Any]:
        """Converts the ChatTurn instance to a dictionary.

        :param pure_text: Whether to encode binary fields to base64. Default is False.
            If True, binary data (e.g. bytes, PIL Image) will be converted
            into base64 strings, but the overall structure of content
            (str or list[dict]) will be preserved so that it can be fully
            restored.
        :type pure_text: bool
        :return: A dictionary representation of the ChatTurn.
        :rtype: dict[str, Any]
        """
        data = {
            "role": self.role,
            "content": self.content,
            "turn_id": self.turn_id,
            "reasoning_content": self.reasoning_content,
            "thinking_blocks": self.thinking_blocks,
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "metadata": dict(self.metadata),
            "strict_mode": self.strict_mode,
        }
        if not pure_text:
            return data

        if isinstance(self.content, str):
            return data

        encoded_content: list[dict[str, Any]] = []
        for part in self.content:
            ctype = part.get("type")
            new_part: dict[str, Any] = dict(part)
            if ctype == "image":
                if "image" in new_part:
                    new_part["image"] = image_to_base64(new_part["image"])
                    new_part["encoding"] = "base64"
            elif ctype in {"pdf", "file", "audio", "video"}:
                if "binary" in new_part:
                    new_part["binary"] = binary_to_base64(new_part["binary"])
                    new_part["encoding"] = "base64"
            encoded_content.append(new_part)
        data["content"] = encoded_content
        return data

    @classmethod
    def from_dict(cls, chat_turn: dict[str, Any], strict_mode: bool = True) -> ChatTurn:
        """Create a ChatTurn from a dictionary.

        :param chat_turn: Dictionary with at least ``role`` and ``content`` keys.
        :type chat_turn: dict[str, Any]
        :param strict_mode: Whether to enforce strict role validation. Defaults to True.
        :type strict_mode: bool
        :return: The constructed ChatTurn instance.
        :rtype: ChatTurn
        :raises ValueError: If ``role`` or ``content`` is missing, or ``content``
            is neither a string nor a list of dicts.
        """
        role = chat_turn.get("role")
        content = chat_turn.get("content")
        strict_mode = chat_turn.get("strict_mode", strict_mode)
        if role is None or content is None:
            raise ValueError("chat_turn must have 'role' and 'content' fields")
        common_kwargs = {
            "role": role,
            "turn_id": chat_turn.get("turn_id"),
            "reasoning_content": chat_turn.get("reasoning_content"),
            "thinking_blocks": chat_turn.get("thinking_blocks"),
            "tool_call_id": chat_turn.get("tool_call_id"),
            "name": chat_turn.get("name"),
            "metadata": dict(chat_turn.get("metadata", {})),
            "strict_mode": strict_mode,
        }
        if isinstance(content, str):
            return cls(content=content, **common_kwargs)
        if not isinstance(content, list):
            raise ValueError("content must be either str or list[ContentPart]")

        restored_content: list[ContentPart] = []
        for part in content:
            ctype = part.get("type")
            encoding = part.get("encoding")
            new_part: dict[str, Any] = dict(part)
            if encoding == "base64":
                if ctype == "image" and "image" in part:
                    new_part["image"] = base64_to_image(part["image"])
                elif ctype in {"pdf", "file", "audio", "video"} and ("binary" in part):
                    new_part["binary"] = base64_to_binary(part["binary"])
                new_part.pop("encoding")
            restored_content.append(cast(ContentPart, new_part))

        return cls(content=restored_content, **common_kwargs)

    def pretty_print(self) -> None:
        header = f"[bold cyan]{self.role.upper()}[/bold cyan]"
        console.print(header)
        summaries: list[str] = []
        if self.role == "tool":
            tool_name = self.name or "unknown"
            tool_call_id = self.tool_call_id or "unknown"
            summaries.append(f"tool={tool_name}")
            summaries.append(f"tool_call_id={tool_call_id}")
        tool_calls = self.tool_calls
        if tool_calls:
            summaries.append(f"tool_calls={len(tool_calls)}")
        if self.reasoning_content or self.thinking_blocks:
            summaries.append("reasoning=present")
        finish_reason = self.metadata.get("finish_reason")
        if finish_reason:
            summaries.append(f"finish_reason={finish_reason}")
        usage = self.metadata.get("usage")
        if isinstance(usage, dict):
            total_tokens = usage.get("total_tokens")
            if total_tokens is not None:
                summaries.append(f"tokens={total_tokens}")
        if summaries:
            console.print(f"[dim]{', '.join(summaries)}[/dim]")
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

    @property
    def tool_calls(self) -> list[ToolCallContentPart]:
        """Returns tool calls exposed inside the content blocks."""
        if isinstance(self.content, str):
            return []
        return [
            cast(ToolCallContentPart, part)
            for part in self.content
            if part.get("type") == "tool_call"
        ]


def _validate_chat_messages(
    chat_messages: list[ChatTurn], info: ValidationInfo
) -> list[ChatTurn]:
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

    if info.data.get("strict_mode", True) is False:
        return chat_messages

    # step 3: check role alternation
    if len(chat_messages) == 0:
        return chat_messages
    for n, turn in enumerate(chat_messages):
        if turn.role == "tool" and not turn.tool_call_id:
            raise ValueError(
                f"The tool chat turn at index {n} must have a tool_call_id."
            )
    if chat_messages[0].role == "system":
        if len(chat_messages) == 1:
            return chat_messages
        if chat_messages[1].role != "user":
            raise ValueError(
                "The role of the chat turn after a system prompt must be 'user', "
                f"but got '{chat_messages[1].role}'"
            )
    elif chat_messages[0].role != "user":
        raise ValueError(
            f"The role of the first chat turn must be 'system' or 'user', "
            f"but got '{chat_messages[0].role}'"
        )

    for n, turn in enumerate(chat_messages[:-1]):
        next_turn = chat_messages[n + 1]
        if turn.role == "system":
            valid_next_roles = {"user"}
        elif turn.role == "user":
            valid_next_roles = {"assistant"}
        elif turn.role == "assistant":
            valid_next_roles = {"tool"} if turn.tool_calls else {"user"}
        elif turn.role == "tool":
            valid_next_roles = {"tool", "assistant"}
        else:
            valid_next_roles = set()
        if next_turn.role not in valid_next_roles:
            raise ValueError(
                f"The role of the chat turn at index {n + 1} must be one of "
                f"{sorted(valid_next_roles)}, but got '{next_turn.role}'"
            )

    if chat_messages[-1].role == "assistant" and chat_messages[-1].tool_calls:
        raise ValueError(
            "An assistant turn with tool calls must be followed by a tool turn."
        )
    return chat_messages


@data(kw_only=True)
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
       (`role="system"`), followed by a user message.
    2. User turns must be followed by assistant turns.
    3. Assistant turns without tool calls must be followed by a user turn or
       terminate the conversation.
    4. Assistant turns with tool calls must be followed by one or more tool
       turns, and tool turns may be followed by another tool turn or an
       assistant turn.

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

    strict_mode: bool = field(default=True, repr=False)
    history: Annotated[
        list[ChatTurn],
        AfterValidator(_validate_chat_messages),
    ] = field(default_factory=list)

    def __getitem__(self, index: int) -> ChatTurn:
        """Returns the chat turn at the specified index."""
        return self.history[index]

    def __setitem__(self, index: int, chat_turn: ChatTurn | dict[str, Any]) -> None:
        """Sets the chat turn at the specified index."""
        if isinstance(chat_turn, dict):
            chat_turn = ChatTurn.from_dict(chat_turn)
        self.history[index] = chat_turn
        return

    def __delitem__(self, index: int) -> None:
        """Deletes the chat turn at the specified index."""
        del self.history[index]
        return

    def insert(self, index: int, chat_turn: ChatTurn | dict[str, Any]) -> None:
        """Inserts a chat turn at the specified index."""
        if isinstance(chat_turn, dict):
            chat_turn = ChatTurn.from_dict(chat_turn)
        self.history.insert(index, chat_turn)
        return

    def __len__(self) -> int:
        """Returns the total number of chat turns in the prompt"""
        return len(self.history)

    def to_list(self, pure_text: bool = False) -> list[dict[str, Any]]:
        """Convert the chat messages to a list of dictionaries.

        :param pure_text: If True, binary content (images, PDFs, etc.) is excluded.
        :type pure_text: bool
        :return: A list of dictionaries, one per chat turn.
        :rtype: list[dict[str, Any]]
        """
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
    def from_list(
        cls, messages: Sequence[ChatTurn | dict[str, Any]], strict_mode: bool = True
    ) -> ChatMessages:
        """Creates a ChatMessages instance from a sequence of dictionaries.
        :param messages: A sequence of dictionaries representing chat turns.
        :type messages: Sequence[ChatTurn | dict[str, Any]]
        :param strict_mode: Whether to enforce strict role validation. Default: True.
            If True, only "user", "assistant", and "system" are allowed as roles.
        :type strict_mode: bool
        :return: An instance of ChatMessages.
        :rtype: ChatMessages
        """
        turns: list[ChatTurn] = []
        for turn in messages:
            if isinstance(turn, dict):
                turns.append(ChatTurn.from_dict(turn, strict_mode=strict_mode))
            else:
                turns.append(turn)
        return cls(history=turns, strict_mode=strict_mode)

    @classmethod
    def from_json(cls, path: str | PathLike, strict_mode: bool = True) -> ChatMessages:
        """Loads the chat messages from a JSON file.

        :param path: The path to the JSON file.
        :type path: str | PathLike
        :return: An instance of ChatMessages.
        :rtype: ChatMessages
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_list(data, strict_mode=strict_mode)

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
        return ChatMessages(history=self.history.copy(), strict_mode=self.strict_mode)

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
