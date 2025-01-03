from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Optional

from transformers import PreTrainedTokenizer

from flexrag.utils import Choices, LOGGER_MANAGER

from .prompt_base import ChatPrompt

TRUNCATION_STRATEGIES = ["left", "right", "history", "demo", "auto"]


logger = LOGGER_MANAGER.get_logger("flexrag.prompt")


class ChatTemplate(ABC):
    @abstractmethod
    def render_to_text(
        self,
        prompt: ChatPrompt,
        add_generation_prompt: bool = True,
    ) -> str:
        return

    @abstractmethod
    def render_to_ids(
        self,
        prompt: ChatPrompt,
        max_length: int = None,
        truncation: Choices(TRUNCATION_STRATEGIES) = "auto",  # type: ignore
        padding: bool = False,
        has_label: bool = False,
        add_generation_prompt: bool = True,
    ):
        return


class HFTemplate(ChatTemplate):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        sys_prompt: str = None,
        chat_template: str = None,  # Jinja template
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.sys_prompt = sys_prompt
        self.chat_template = chat_template
        return

    def render_to_text(
        self,
        prompt: ChatPrompt,
        add_generation_prompt: bool = True,
    ) -> str:
        # add default system prompt
        prompt_ = prompt.to_list()
        if (len(prompt_) == 0) and (self.sys_prompt is not None):
            prompt_.append({"role": "system", "content": self.sys_prompt})
        if (prompt_[0]["role"] != "system") and (self.sys_prompt is not None):
            prompt_.insert(0, {"role": "system", "content": self.sys_prompt})
        # apply template
        prefix = self.tokenizer.apply_chat_template(
            prompt_,
            tokenize=False,
            chat_template=self.chat_template,
            add_generation_prompt=add_generation_prompt,
        )
        return prefix

    def render_to_ids(
        self,
        prompt: ChatPrompt,
        max_length: Optional[int] = None,
        truncation: Choices(TRUNCATION_STRATEGIES) = "auto",  # type: ignore
        padding: bool = False,
        has_label: bool = False,  # TODO: support label(mask_input)
        add_generation_prompt: bool = True,
    ) -> list[int] | tuple[list[int], list[int]]:
        def _encode(
            prompt: ChatPrompt,
            max_length: int = None,
            truncation: bool = False,
            truncation_side: str = "left",
        ) -> list[int]:
            prefix = self.render_to_text(prompt, add_generation_prompt)
            self.tokenizer.truncation_side = truncation_side
            ids = self.tokenizer(
                prefix,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
            )["input_ids"]
            return ids

        # truncate the prompt
        prompt = deepcopy(prompt)
        if max_length is not None:
            match truncation:
                case "left":
                    ids = _encode(
                        prompt,
                        max_length=max_length,
                        truncation=True,
                        truncation_side="left",
                    )
                case "right":
                    ids = _encode(
                        prompt,
                        max_length=max_length,
                        truncation=True,
                        truncation_side="right",
                    )
                case "history":
                    pre_ids = _encode(prompt)
                    while len(pre_ids) > max_length:
                        if len(prompt.history) > 1:
                            prompt.pop_history(0)
                            prompt.pop_history(0)
                        else:
                            raise ValueError(
                                "Unable to truncate the prompt using `history` strategy."
                            )
                        pre_ids = _encode(prompt)
                    ids = pre_ids
                case "demo":
                    pre_ids = _encode(prompt)
                    while len(pre_ids) > max_length:
                        if len(prompt.demonstrations) > 0:
                            prompt.pop_demonstration(0)
                        else:
                            raise ValueError(
                                "Unable to truncate the prompt using `demo` strategy."
                            )
                case "auto":
                    pre_ids = _encode(prompt)
                    while len(pre_ids) > max_length:
                        if len(prompt.history) > 1:
                            prompt.pop_history(0)
                            prompt.pop_history(0)
                            pre_ids = _encode(prompt)
                        elif len(prompt.demonstrations) > 0:
                            prompt.pop_demonstration(0)
                            pre_ids = _encode(prompt)
                        else:
                            pre_ids = _encode(
                                prompt, truncation=True, truncation_side="left"
                            )
                    ids = pre_ids
                case _:
                    raise ValueError("Unsupported truncation strategy.")
        else:
            ids = _encode(prompt)
        return ids


# register default templates
Llama3Template = partial(
    HFTemplate,
    sys_prompt="You are a pirate chatbot who always responds in pirate speak!",
)

Qwen2Template = partial(
    HFTemplate,
    sys_prompt="You are a helpful assistant.",
)

Phi3Template = HFTemplate


def load_template(
    tokenizer: PreTrainedTokenizer,
    model_name: Optional[str] = None,
) -> ChatTemplate:
    """
    Load ChatTemplate for different models. If model_name is not provided, the default template in the Tokenizer will be used.

    :param tokenizer: The tokenizer used to encode the prompt.
    :param model_name: The name of the model. Default is None.
    :type tokenizer: PreTrainedTokenizer
    :type model_name: Optional[str]
    :return: The loaded ChatTemplate
    :rtype: ChatTemplate
    """
    if model_name is None:
        logger.warning("model_name is not provided, using default template.")
        return HFTemplate(tokenizer=tokenizer)
    if "Phi-3" in model_name:
        return Phi3Template(tokenizer=tokenizer)
    elif "Meta-Llama-3" in model_name:
        return Llama3Template(tokenizer=tokenizer)
    elif "Qwen/Qwen2" in model_name:
        return Qwen2Template(tokenizer=tokenizer)
    raise ValueError(f"Unsupported architecture: {model_name}")
