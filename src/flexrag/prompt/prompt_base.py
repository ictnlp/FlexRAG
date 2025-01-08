import base64
import json
from dataclasses import dataclass, field
from io import BytesIO
from os import PathLike
from typing import Optional

from PIL.Image import Image

from flexrag.utils import Choices


@dataclass
class ChatTurn:
    role: Choices(["user", "assistant", "system"])  # type: ignore
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, chat_turn: dict[str, str]):
        return cls(role=chat_turn["role"], content=chat_turn["content"])


@dataclass
class MultiModelChatTurn:
    role: Choices(["user", "assistant", "system"])  # type: ignore
    content: list[dict[str, str | Image]] | str

    def to_dict(self, encode_img: bool = True) -> dict[str, str | dict]:
        if not encode_img:
            return {"role": self.role, "content": self.content}

        # encode the image into base64 string
        new_content = []
        for content in self.content:
            if isinstance(content, str):
                new_content.append(content)
            elif content["type"] == "text":
                new_content.append(content)
            elif content["type"] == "image":
                assert isinstance(content["image"], Image)
                image_buffer = BytesIO()
                content["image"].save(image_buffer, format="PNG")
                image_bytes = image_buffer.getvalue()
                image_string = base64.b64encode(image_bytes).decode("utf-8")
                new_content.append({"type": "image", "image": image_string})
            else:
                raise ValueError("Invalid content type")
        return {"role": self.role, "content": new_content}

    @classmethod
    def from_dict(cls, chat_turn: dict[str, str | dict]):
        # load the image data
        loaded_content = []
        for content in chat_turn["content"]:
            if isinstance(content, str):
                loaded_content.append(content)
            elif content["type"] == "text":
                loaded_content.append(content)
            elif content["type"] == "image":
                if isinstance(content["image"], Image):
                    loaded_content.append(content)
                elif isinstance(content["image"], str):
                    image_bytes = base64.b64decode(content["image"])
                    image_buffer = BytesIO(image_bytes)
                    image = Image.open(image_buffer)
                    loaded_content.append({"type": "image", "image": image})
                else:
                    raise ValueError("Invalid image content")
            else:
                raise ValueError("Invalid content type")
        return cls(role=chat_turn["role"], content=chat_turn["content"])


@dataclass
class ChatPrompt:
    system: Optional[ChatTurn] = None
    history: list[ChatTurn] = field(default_factory=list)
    demonstrations: list[list[ChatTurn]] = field(default_factory=list)

    def __init__(
        self,
        system: Optional[str | ChatTurn] = None,
        history: list[ChatTurn] | list[dict[str, str]] = [],
        demonstrations: list[list[ChatTurn]] | list[list[dict[str, str]]] = [],
    ):
        # set system
        if isinstance(system, str):
            system = ChatTurn(role="system", content=system)
        self.system = system

        # set history
        if len(history) > 0:
            if isinstance(history[0], dict):
                history = [ChatTurn.from_dict(turn) for turn in history]
        self.history = history

        # set demonstrations
        if len(demonstrations) > 0:
            if isinstance(demonstrations[0][0], dict):
                demonstrations = [
                    [ChatTurn.from_dict(turn) for turn in demo]
                    for demo in demonstrations
                ]
        self.demonstrations = demonstrations
        return

    def to_list(self) -> list[dict[str, str]]:
        data = []
        if self.system is not None:
            data.append({"role": "system", "content": self.system.content})
        for demo in self.demonstrations:
            for turn in demo:
                data.append(turn.to_dict())
        for turn in self.history:
            data.append(turn.to_dict())
        return data

    def to_json(self, path: str | PathLike):
        data = {"system": self.system.to_dict(), "history": [], "demonstrations": []}
        for turn in self.history:
            data["history"].append(turn.to_dict())
        for demo in self.demonstrations:
            data["demonstrations"].append([turn.to_dict() for turn in demo])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    @classmethod
    def from_list(cls, prompt: list[dict[str, str]]) -> "ChatPrompt":
        history = [ChatTurn.from_dict(turn) for turn in prompt]
        if history[0].role == "system":
            system = history.pop(0)
        else:
            system = None
        return cls(system=system, history=history, demonstrations=[])

    @classmethod
    def from_json(cls, path: str | PathLike) -> "ChatPrompt":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_list(data)
        return cls(
            system=ChatTurn.from_dict(data["system"]),
            history=[ChatTurn.from_dict(turn) for turn in data["history"]],
            demonstrations=[
                [ChatTurn.from_dict(turn) for turn in demo]
                for demo in data["demonstrations"]
            ],
        )

    def load_demonstrations(self, demo_path: str | PathLike):
        with open(demo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.demonstrations = [
            [ChatTurn.from_dict(turn) for turn in demo] for demo in data
        ]
        return

    def pop_history(self, n: int) -> ChatTurn:
        return self.history.pop(n)

    def pop_demonstration(self, n: int) -> list[ChatTurn]:
        return self.demonstrations.pop(n)

    def update(
        self, chat_turns: ChatTurn | dict[str, str] | list[ChatTurn | dict[str, str]]
    ) -> None:
        if not isinstance(chat_turns, list):
            chat_turns = [chat_turns]
        for chat_turn in chat_turns:
            if isinstance(chat_turn, dict):
                chat_turn = ChatTurn.from_dict(chat_turn)
            self.history.append(chat_turn)
        return

    def clean(self, clean_system: bool = False):
        if clean_system:
            self.system = None
        self.history = []
        self.demonstrations = []
        return

    def __len__(self) -> int:
        system_num = 0 if self.system is None else 1
        history_num = len(self.history)
        demo_num = sum([len(demo) for demo in self.demonstrations])
        return system_num + history_num + demo_num


@dataclass
class MultiModelChatPrompt:
    """
    This class shares almost all the methods with ChatPrompt.
    However, the Generics in Python does not support calling the TypeVar's classmethod.
    So we have to duplicate the code here.
    """

    system: Optional[MultiModelChatTurn] = None
    history: list[MultiModelChatTurn] = field(default_factory=list)
    demonstrations: list[list[MultiModelChatTurn]] = field(default_factory=list)

    def __init__(
        self,
        system: Optional[str | MultiModelChatTurn] = None,
        history: list[MultiModelChatTurn] | list[dict[str, str]] = [],
        demonstrations: (
            list[list[MultiModelChatTurn]] | list[list[dict[str, str]]]
        ) = [],
    ):
        # set system
        if isinstance(system, str):
            system = MultiModelChatTurn(role="system", content=system)
        self.system = system

        # set history
        if len(history) > 0:
            if isinstance(history[0], dict):
                history = [MultiModelChatTurn.from_dict(turn) for turn in history]
        self.history = history

        # set demonstrations
        if len(demonstrations) > 0:
            if isinstance(demonstrations[0][0], dict):
                demonstrations = [
                    [MultiModelChatTurn.from_dict(turn) for turn in demo]
                    for demo in demonstrations
                ]
        self.demonstrations = demonstrations
        return

    def to_list(self) -> list[dict[str, str]]:
        data = []
        if self.system is not None:
            data.append({"role": "system", "content": self.system.content})
        for demo in self.demonstrations:
            for turn in demo:
                data.append(turn.to_dict())
        for turn in self.history:
            data.append(turn.to_dict())
        return data

    def to_json(self, path: str | PathLike):
        data = {"system": self.system.to_dict(), "history": [], "demonstrations": []}
        for turn in self.history:
            data["history"].append(turn.to_dict())
        for demo in self.demonstrations:
            data["demonstrations"].append([turn.to_dict() for turn in demo])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        return

    @classmethod
    def from_list(cls, prompt: list[dict[str, str]]) -> "ChatPrompt":
        history = [MultiModelChatTurn.from_dict(turn) for turn in prompt]
        if history[0].role == "system":
            system = history.pop(0)
        else:
            system = None
        return cls(system=system, history=history, demonstrations=[])

    @classmethod
    def from_json(cls, path: str | PathLike) -> "ChatPrompt":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return cls.from_list(data)
        return cls(
            system=MultiModelChatTurn.from_dict(data["system"]),
            history=[MultiModelChatTurn.from_dict(turn) for turn in data["history"]],
            demonstrations=[
                [MultiModelChatTurn.from_dict(turn) for turn in demo]
                for demo in data["demonstrations"]
            ],
        )

    @property
    def images(self) -> list[Image]:
        images = []
        for turn in self.history:
            if isinstance(turn.content, list):
                for content in turn.content:
                    if content["type"] == "image":
                        images.append(content["image"])
        return images

    def load_demonstrations(self, demo_path: str | PathLike):
        with open(demo_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.demonstrations = [
            [MultiModelChatTurn.from_dict(turn) for turn in demo] for demo in data
        ]
        return

    def pop_history(self, n: int) -> MultiModelChatTurn:
        return self.history.pop(n)

    def pop_demonstration(self, n: int) -> list[MultiModelChatTurn]:
        return self.demonstrations.pop(n)

    def update(
        self,
        chat_turns: (
            MultiModelChatTurn
            | dict[str, str]
            | list[MultiModelChatTurn | dict[str, str]]
        ),
    ) -> None:
        if not isinstance(chat_turns, list):
            chat_turns = [chat_turns]
        for chat_turn in chat_turns:
            if isinstance(chat_turn, dict):
                chat_turn = MultiModelChatTurn.from_dict(chat_turn)
            self.history.append(chat_turn)
        return

    def clean(self, clean_system: bool = False):
        if clean_system:
            self.system = None
        self.history = []
        self.demonstrations = []
        return

    def __len__(self) -> int:
        system_num = 0 if self.system is None else 1
        history_num = len(self.history)
        demo_num = sum([len(demo) for demo in self.demonstrations])
        return system_num + history_num + demo_num
