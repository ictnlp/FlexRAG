import asyncio
from abc import abstractmethod

from flexrag.common import configure
from flexrag.common.async_utils import BackgroundEventLoop
from flexrag.common.dataclasses import ChatMessages, ChatTurn

from .generator_base import GenerationConfig, GeneratorBase


@configure
class APIBasedGeneratorBaseConfig:
    max_concurrency: int = 1


class APIBasedGeneratorBase(GeneratorBase):
    """Base class for API-based LLM generators.

    The APIBasedGeneratorBase uses a background event loop to run async calls,
    and provides both async and sync interfaces with concurrency control.
    It manages a background event loop thread to execute asynchronous tasks and uses
    an asyncio Semaphore to limit the maximum number of concurrent API requests.

    This class provides the following public methods:
    - :meth:`chat`: Synchronous chat interface.
    - :meth:`async_chat`: Asynchronous chat interface.
    - :meth:`generate`: Synchronous generation interface.
    - :meth:`async_generate`: Asynchronous generation interface.

    The subclasses should implement the following methods:

        >>> async def _create_client(self, config: APIBasedGeneratorBaseConfig):
        >>>     # Create and return the async client instance.
        >>>     ...

        >>> async def _async_chat_impl(
        >>>     self,
        >>>     client,
        >>>     messages: ChatMessages,
        >>>     generation_config: GenerationConfig,
        >>> ) -> ChatTurn:
        >>>     # Perform the async chat call using the client.
        >>>     ...

        >>> async def _async_generate_impl(
        >>>     self,
        >>>     client,
        >>>     prompt: str,
        >>>     generation_config: GenerationConfig,
        >>> ) -> str:
        >>>     # Perform the async generate call using the client.
        >>>     ...
    """

    def __init__(self, config: APIBasedGeneratorBaseConfig):
        self._loop_thread = BackgroundEventLoop()
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        self._client = None  # Will be created lazily
        self._config = config
        return

    async def _get_async_client(self):
        """Create client lazily inside background event loop."""
        if self._client is None:
            self._client = await self._create_client(self._config)
        return self._client

    @abstractmethod
    async def _create_client(self, config: APIBasedGeneratorBaseConfig):
        """Implemented by subclasses, create and return the async client instance."""
        return

    @abstractmethod
    async def _async_chat_impl(
        self,
        client,
        messages: ChatMessages,
        generation_config: GenerationConfig,
    ) -> ChatTurn:
        """Implemented by subclasses, perform the async chat call."""
        return

    @abstractmethod
    async def _async_generate_impl(
        self,
        client,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> str:
        """Implemented by subclasses, perform the async generate call."""
        return

    async def _async_chat_core(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        # Normalize input to list of ChatMessages
        if isinstance(messages, ChatMessages) or isinstance(messages[0], dict):
            messages = [messages]
        for i in range(len(messages)):
            if isinstance(messages[i], list):
                messages[i] = ChatMessages.from_list(messages[i])
        # Process each chat request with concurrency control
        sample_num = generation_config.sample_num if generation_config else 1
        async with self._semaphore:
            client = await self._get_async_client()
            tasks = [
                self._async_chat_impl(client, msg, generation_config)
                for msg in messages
                for _ in range(sample_num)
            ]
            flat_response = await asyncio.gather(*tasks)
        responses: list[list[ChatTurn]] = [
            flat_response[i * sample_num : (i + 1) * sample_num]
            for i in range(len(messages))
        ]
        return responses

    async def _async_generate_core(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        # Normalize input to list of strings
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        # Process each generation request with concurrency control
        sample_num = generation_config.sample_num if generation_config else 1
        async with self._semaphore:
            client = await self._get_async_client()
            tasks = [
                self._async_generate_impl(client, prefix, generation_config)
                for prefix in prefixes
                for _ in range(sample_num)
            ]
            flat_response = await asyncio.gather(*tasks)
        responses: list[list[str]] = [
            flat_response[i * sample_num : (i + 1) * sample_num]
            for i in range(len(prefixes))
        ]
        return responses

    async def async_chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        """Asynchronously chat with the model.

        :param messages: A list of ChatMessages or a single ChatMessages object.
            Each ChatMessages object represents a conversation history.
            If a list of dicts is provided, it will be converted to ChatMessages.
        :type messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict]
        :param generation_config: Configuration for generation, defaults to None.
        :type generation_config: GenerationConfig | None, optional
        :return: A list of lists of ChatTurn objects.
            The outer list corresponds to the input messages, and the inner list contains
            the generated responses (multiple samples if configured).
        :rtype: list[list[ChatTurn]]
        """
        thread_future = self._loop_thread.run_async(
            self._async_chat_core(messages, generation_config)
        )
        return await asyncio.wrap_future(thread_future)

    async def async_generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        """Asynchronously generate text from the model.

        :param prefixes: A list of prompt strings or a single prompt string.
        :type prefixes: list[str] | str
        :param generation_config: Configuration for generation, defaults to None.
        :type generation_config: GenerationConfig | None, optional
        :return: A list of lists of generated strings.
            The outer list corresponds to the input prefixes, and the inner list contains
            the generated outputs (multiple samples if configured).
        :rtype: list[list[str]]
        """
        thread_future = self._loop_thread.run_async(
            self._async_generate_core(prefixes, generation_config)
        )
        return await asyncio.wrap_future(thread_future)

    def chat(
        self,
        messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict],
        generation_config: GenerationConfig | None = None,
    ) -> list[list[ChatTurn]]:
        """Chat with the model.

        :param messages: A list of ChatMessages or a single ChatMessages object.
            Each ChatMessages object represents a conversation history.
            If a list of dicts is provided, it will be converted to ChatMessages.
        :type messages: list[ChatMessages] | list[list[dict]] | ChatMessages | list[dict]
        :param generation_config: Configuration for generation, defaults to None.
        :type generation_config: GenerationConfig | None, optional
        :return: A list of lists of ChatTurn objects.
            The outer list corresponds to the input messages, and the inner list contains
            the generated responses (multiple samples if configured).
        :rtype: list[list[ChatTurn]]
        """
        future = self._loop_thread.run_async(
            self._async_chat_core(messages, generation_config)
        )
        return future.result()

    def generate(
        self,
        prefixes: list[str] | str,
        generation_config: GenerationConfig | None = None,
    ) -> list[list[str]]:
        """Generate text from the model.

        :param prefixes: A list of prompt strings or a single prompt string.
        :type prefixes: list[str] | str
        :param generation_config: Configuration for generation, defaults to None.
        :type generation_config: GenerationConfig | None, optional
        :return: A list of lists of generated strings.
            The outer list corresponds to the input prefixes, and the inner list contains
            the generated outputs (multiple samples if configured).
        :rtype: list[list[str]]
        """
        future = self._loop_thread.run_async(
            self._async_generate_core(prefixes, generation_config)
        )
        return future.result()
