from abc import ABC, abstractmethod
from dataclasses import field
from typing import Optional

from flexrag.common import LOGGER_MANAGER, ChatMessages, ChatTurn, Register, configure

logger = LOGGER_MANAGER.get_logger("flexrag.models")


@configure
class GenerationConfig:
    """Configuration for text generation.
    Note that not all options are supported by all models.

    :param do_sample: Whether to use sampling for generation. Defaults to True.
    :type do_sample: bool
    :param sample_num: The number of samples to generate. Defaults to 1.
    :type sample_num: int
    :param temperature: The temperature of the sampling distribution. Defaults to 1.0.
    :type temperature: float
    :param max_new_tokens: The maximum number of tokens to generate. Defaults to None.
        None means no limit.
    :type max_new_tokens: Optional[int]
    :param top_p: The cumulative probability for nucleus sampling. Defaults to None.
    :type top_p: Optional[float]
    :param top_k: The number of tokens to consider for top-k sampling. Defaults to None.
    :type top_k: Optional[int]
    :param eos_token_id: The token id for the end of sentence token. Defaults to None.
    :type eos_token_id: Optional[int]
    :param stop_str: A list of strings to stop generation. Defaults to [].
    :type stop_str: list[str]
    """

    do_sample: bool = True
    sample_num: int = 1
    temperature: float = 1.0
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_str: list[str] = field(default_factory=list)

    def __post_init__(self):
        # check values
        assert self.sample_num > 0, "sample_num must be greater than 0"
        if self.sample_num > 1:
            assert self.do_sample, "do_sample must be True when sample_num > 1"
        assert self.temperature >= 0, "temperature must be greater than or equal to 0"
        if self.max_new_tokens is not None:
            assert self.max_new_tokens > 0, "max_new_tokens must be greater than 0"
        if self.top_p is not None:
            assert 0 <= self.top_p <= 1, "top_p must be between 0 and 1"
        if self.top_k is not None:
            assert self.top_k > 0, "top_k must be greater than 0"


class GeneratorBase(ABC):
    """Base class for generators.
    The generator can generate text or chat responses based on the given prefixes or messages.
    The subclasses must implement the `chat` and `generate` methods.
    """

    @abstractmethod
    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        """chat with the model using model templates.

        :param messages: A batch of ChatMessages.
        :type messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict]
        :param generation_config: GenerationConfig. Defaults to None.
        :type generation_config: GenerationConfig | None
        :return: A batch of chat responses.
        :rtype: list[list[ChatTurn]]
        """
        return

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        """The async version of chat.

        :param messages: A batch of ChatMessages.
        :type messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict]
        :param generation_config: GenerationConfig. Defaults to None.
        :type generation_config: GenerationConfig | None
        :return: A batch of chat responses.
        :rtype: list[list[ChatTurn]]
        """
        logger.warning(
            "Current model does not support asynchronous chat,"
            " thus the code will be run in synchronous mode"
        )
        return self.chat(messages=messages, generation_config=generation_config)

    @abstractmethod
    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        """generate text with the model using the given prefixes.

        :param prefixes: A batch of prefixes.
        :type prefixes: list[str] | str
        :param generation_config: GenerationConfig. Defaults to None.
        :type generation_config: GenerationConfig | None
        :return: A batch of generated text.
        :rtype: list[list[str]]
        """
        return

    async def async_generate(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        """The async version of generate.

        :param prefixes: A batch of prefixes.
        :type prefixes: list[str] | str
        :param generation_config: GenerationConfig. Defaults to None.
        :type generation_config: GenerationConfig | None
        :return: A batch of generated text.
        :rtype: list[list[str]]
        """
        logger.warning(
            "Current generator does not support asynchronous generate,"
            " thus the code will be run in synchronous mode"
        )
        return self.generate(prefixes=prefixes, generation_config=generation_config)


GENERATORS = Register[GeneratorBase]("generator")
