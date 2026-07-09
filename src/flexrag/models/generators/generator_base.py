import asyncio
from abc import ABC, abstractmethod
from dataclasses import field
from typing import Any, Optional, Protocol, TypeAlias

from flexrag.common import ChatMessages, ChatTurn, Register, configure

GeneratorPrefixes: TypeAlias = str | list[str]
GeneratorMessages: TypeAlias = (
    ChatMessages
    | list[dict[str, Any]]
    | list[ChatMessages]
    | list[list[dict[str, Any]]]
)


def _normalize_generation_prefixes(prefixes: GeneratorPrefixes) -> list[str]:
    if isinstance(prefixes, str):
        return [prefixes]
    for prefix in prefixes:
        if not isinstance(prefix, str):
            raise TypeError("Generator prefixes must be strings.")
    return prefixes


def _normalize_chat_messages(messages: GeneratorMessages) -> list[ChatMessages]:
    if isinstance(messages, ChatMessages):
        return [messages]
    if not messages:
        return []
    if isinstance(messages[0], dict):
        return [ChatMessages.from_list(messages)]

    normalized: list[ChatMessages] = []
    for message in messages:
        if isinstance(message, ChatMessages):
            normalized.append(message)
        else:
            normalized.append(ChatMessages.from_list(message))
    return normalized


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
    :param tools: Provider-native tool definitions passed through to supported chat models.
        Defaults to [].
    :type tools: list[dict[str, Any]]
    :param reasoning_effort: Provider-specific reasoning effort hint. Defaults to None.
    :type reasoning_effort: Optional[str]
    :param response_format: OpenAI compatible schema constraint.
        Defaults to None.
    :type response_format: Optional[dict[str, Any]]
    """

    do_sample: bool = True
    sample_num: int = 1
    temperature: float = 1.0
    max_new_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_token_id: Optional[int] = None
    stop_str: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    reasoning_effort: Optional[str] = None
    response_format: Optional[dict[str, Any]] = None

    def __post_init__(self):
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


class GeneratorProtocol(Protocol):
    """Protocol for directly usable raw generators.

    Raw generators expose a common canonical-batch interface for direct use.
    Implementations do not provide runtime policies such as deployment
    batching, progress logging, process isolation, retry, or rate limiting.
    """

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        ...

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        ...

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate completions for each input prefix.
        """
        ...

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions asynchronously for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate completions for each input prefix.
        """
        ...


class LocalGeneratorBase(ABC):
    """Thin base class for directly usable local generators.

    Subclasses implement synchronous canonical-batch ``_generate_batch`` and
    ``_chat_batch``. The public methods split direct-use calls according to
    ``batch_size`` and merge the resulting batches. The async methods are
    convenience wrappers built with ``asyncio.to_thread``; they keep an event
    loop responsive but do not provide process isolation, retry, rate limiting,
    progress logging, or true Python-level parallelism.
    """

    def __init__(self, batch_size: int = 1) -> None:
        """Initialize direct-use local generator batching.

        :param batch_size: Maximum batch size used by the raw local generator's
            public ``generate`` and ``chat`` methods.
        :raises ValueError: If ``batch_size`` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0.")
        self.batch_size = batch_size
        return

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        normalized_messages = _normalize_chat_messages(messages)
        resolved_batch_size = batch_size or self.batch_size
        results: list[list[ChatTurn]] = []
        for i in range(0, len(normalized_messages), resolved_batch_size):
            results.extend(
                self._chat_batch(
                    normalized_messages[i : i + resolved_batch_size],
                    generation_config=generation_config,
                )
            )
        return results

    @abstractmethod
    def _chat_batch(
        self,
        messages: list[ChatMessages],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses for one implementation batch.

        :param messages: Normalized chat conversations to process.
        :param generation_config: Optional generation options for this call.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        return

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        return await asyncio.to_thread(
            self.chat,
            messages,
            generation_config=generation_config,
            batch_size=batch_size,
        )

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate completions for each input prefix.
        """
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        resolved_batch_size = batch_size or self.batch_size
        results: list[list[str]] = []
        for i in range(0, len(normalized_prefixes), resolved_batch_size):
            results.extend(
                self._generate_batch(
                    normalized_prefixes[i : i + resolved_batch_size],
                    generation_config=generation_config,
                )
            )
        return results

    @abstractmethod
    def _generate_batch(
        self,
        prefixes: list[str],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        """Generate text completions for one implementation batch.

        :param prefixes: Text prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :return: One list of candidate completions for each input prefix.
        """
        return

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions asynchronously for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Optional per-call batch size override.
        :return: One list of candidate completions for each input prefix.
        """
        return await asyncio.to_thread(
            self.generate,
            prefixes,
            generation_config=generation_config,
            batch_size=batch_size,
        )


class RemoteGeneratorBase(ABC):
    """Thin base class for directly usable remote generators.

    Subclasses implement single-sample async core methods. The public async
    batch methods call those cores sequentially for direct use. The synchronous
    methods run the async batch methods with ``asyncio.run`` and must not be
    called from an already running event loop. ``batch_size`` is accepted by
    public methods for protocol compatibility and ignored.
    """

    @staticmethod
    def _ensure_sync_bridge_allowed(method_name: str) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError(
            f"{method_name} cannot be called from a running event loop. "
            f"Use async_{method_name} instead."
        )

    @abstractmethod
    async def _async_chat_one(
        self,
        messages: ChatMessages,
        generation_config: GenerationConfig | None = None,
    ) -> list[ChatTurn]:
        """Generate chat responses for one conversation.

        :param messages: Normalized chat conversation to process.
        :param generation_config: Optional generation options for this call.
        :return: Candidate assistant turns for the input conversation.
        """
        return

    @abstractmethod
    async def _async_generate_one(
        self,
        prefix: str,
        generation_config: GenerationConfig | None = None,
    ) -> list[str]:
        """Generate text completions for one prefix.

        :param prefix: Text prefix to continue.
        :param generation_config: Optional generation options for this call.
        :return: Candidate completions for the input prefix.
        """
        return

    def chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses synchronously for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Accepted for protocol compatibility and ignored.
        :return: One list of candidate assistant turns for each input
            conversation.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("chat")
        del batch_size
        return asyncio.run(self.async_chat(messages, generation_config))

    async def async_chat(
        self,
        messages: GeneratorMessages,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[ChatTurn]]:
        """Generate chat responses asynchronously for one conversation or a batch.

        :param messages: Chat messages or message dictionaries to process.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Accepted for protocol compatibility and ignored.
        :return: One list of candidate assistant turns for each input
            conversation.
        """
        del batch_size
        normalized_messages = _normalize_chat_messages(messages)
        return [
            await self._async_chat_one(message, generation_config)
            for message in normalized_messages
        ]

    def generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions synchronously for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Accepted for protocol compatibility and ignored.
        :return: One list of candidate completions for each input prefix.
        :raises RuntimeError: If called from a running event loop.
        """
        self._ensure_sync_bridge_allowed("generate")
        del batch_size
        return asyncio.run(self.async_generate(prefixes, generation_config))

    async def async_generate(
        self,
        prefixes: GeneratorPrefixes,
        generation_config: GenerationConfig | None = None,
        *,
        batch_size: int | None = None,
    ) -> list[list[str]]:
        """Generate text completions asynchronously for one prefix or a batch.

        :param prefixes: Text prefix or prefixes to continue.
        :param generation_config: Optional generation options for this call.
        :param batch_size: Accepted for protocol compatibility and ignored.
        :return: One list of candidate completions for each input prefix.
        """
        del batch_size
        normalized_prefixes = _normalize_generation_prefixes(prefixes)
        return [
            await self._async_generate_one(prefix, generation_config)
            for prefix in normalized_prefixes
        ]


GENERATORS = Register[GeneratorProtocol]("generator")
