from dataclasses import field
from typing import Annotated, Any, Optional

from flexrag.common import LOGGER_MANAGER, Choices, configure, data
from flexrag.common.dataclasses import ChatMessages, RetrievedContext
from flexrag.context_refine import REFINERS, RefinerConfig
from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.ranker import RANKERS, RankerConfig
from flexrag.retrievers import RETRIEVERS, RetrieverConfig

from .assistant import ASSISTANTS, AssistantBase, AssistantResponse

logger = LOGGER_MANAGER.get_logger("flexrag.assistant.modular")


@data
class SearchResult:
    """The dataclass for saving the results of a search operation.

    :param query: The query for the search. Required.
    :type query: str
    :param contexts: The contexts retrieved for the query. Required.
    :type contexts: list[RetrievedContext]
    :param metadata: Additional metadata about the search result. Defaults to None.
    :type metadata: Optional[dict[str, Any]]
    """

    query: str
    contexts: list[RetrievedContext]
    metadata: Optional[dict[str, Any]] = None


@configure
class ModularAssistantConfig(
    GeneratorConfig, GenerationConfig, RetrieverConfig, RankerConfig, RefinerConfig
):
    """The configuration for the modular assistant.

    :param used_fields: The fields to use in the context. Defaults to [].
    :type used_fields: list[str], optional
    :param context_incorporation: How to incorporate context into the messages.
    Defaults to "user_prompt". Available options are:

        * "system_prompt": Incorporate context into the system prompt.
        * "user_prompt": Incorporate context into the last user prompt.

    :type context_incorporation: str
    """

    used_fields: list[str] = field(default_factory=list)
    context_incorporation: Annotated[
        str,
        Choices("system_prompt", "user_prompt"),
    ] = "user_prompt"


@ASSISTANTS("modular", config_class=ModularAssistantConfig)
class ModularAssistant(AssistantBase):
    """The modular RAG assistant that supports retrieval, reranking, and generation."""

    def __init__(self, cfg: ModularAssistantConfig):
        # set basic args
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1
        self.used_fields = cfg.used_fields
        self.context_incorporation = cfg.context_incorporation

        # load generator
        self.generator = GENERATORS.load(cfg)
        assert self.generator is not None, "Generator is not loaded."

        # load retriever
        self.retriever = RETRIEVERS.load(cfg)

        # load ranker
        self.reranker = RANKERS.load(cfg)

        # load refiners
        self.refiner = REFINERS.load(cfg)
        return

    def answer(
        self,
        messages: ChatMessages | list[dict],
        disable_retrieval: bool = False,
    ) -> AssistantResponse:
        if isinstance(messages, list):
            messages = ChatMessages.from_list(messages)
        if not disable_retrieval:
            ctxs, history = self.search(messages[-1].content)
            response = self.answer_with_contexts(messages, ctxs)
        else:
            ctxs = []
            history = []
            response = self.answer_with_contexts(messages)
        response.metadata["search_histories"] = history
        return response

    def search(self, query: str) -> tuple[list[RetrievedContext], list[SearchResult]]:
        """Search for relevant contexts based on the query.

        :param query: The query to search for.
        :type query: str
        :return: A tuple containing:
            - A list of retrieved contexts.
            - A list of SearchResult, each containing the query and the contexts retrieved.
        :rtype: tuple[list[RetrievedContext], list[SearchResult]]
        """
        if self.retriever is None:
            return [], []
        # searching for contexts
        search_histories = []
        ctxs = self.retriever.search(query=[query])[0]
        search_histories.append(SearchResult(query=f"search: {query}", contexts=ctxs))

        # reranking
        if self.reranker is not None:
            results = self.reranker.rank(query, ctxs)
            ctxs = results.candidates
            search_histories.append(
                SearchResult(query=f"rerank: {query}", contexts=ctxs)
            )

        # refine
        for refiner in self.refiner:
            ctxs = refiner.refine(ctxs)
            search_histories.append(
                SearchResult(query=f"refine: {query}", contexts=ctxs)
            )

        return ctxs, search_histories

    def answer_with_contexts(
        self,
        messages: ChatMessages | list[dict],
        contexts: list[RetrievedContext] = [],
    ) -> AssistantResponse:
        # convert messages to ChatMessages if it's a list of dicts
        if isinstance(messages, list):
            messages = ChatMessages.from_list(messages)

        # if no contexts, generate response without context
        if len(contexts) == 0:
            response = self.generator.chat([messages], generation_config=self.gen_cfg)
            return AssistantResponse(
                response=response[0][0],
                contexts=contexts,
                metadata={"prompt": messages},
            )

        # concatenate contexts into a string
        context_str = ""
        for n, context in enumerate(contexts):
            if len(self.used_fields) == 0:
                ctx = ""
                for field_name, field_value in context.data.items():
                    ctx += f"{field_name}: {field_value}\n"
            elif len(self.used_fields) == 1:
                ctx = context.data[self.used_fields[0]]
            else:
                ctx = ""
                for field_name in self.used_fields:
                    ctx += f"{field_name}: {context.data[field_name]}\n"
            context_str += f"Context {n + 1}: {ctx}\n\n"

        # incorporate context into messages
        prompt = messages.copy()
        match self.context_incorporation:
            case "system_prompt":
                if prompt.system is not None:
                    prompt.system = (
                        f"{prompt.system}\n\n"
                        "Here are some context documents that may be relevant to this conversation:\n\n"
                        f"{context_str}"
                    )
                else:
                    prompt.system = (
                        "You are a helpful and knowledgeable AI assistant. "
                        "You may be provided with one or more context documents alongside user messages. "
                        "These documents may or may not be relevant to the current query."
                        "If unsure about the relevance or completeness of contexts, "
                        "please answer based on your own knowledge."
                        "Here are some context documents that may be relevant to this conversation:\n\n"
                        f"{context_str}"
                    )
            case "user_prompt":
                prompt[-1].content = (
                    f"Here are some context documents that may be relevant to this conversation:\n\n"
                    f"{context_str}\n"
                    f"{prompt[-1].content}"
                )
            case _:
                raise ValueError(
                    f"Unknown context incorporation method: {self.context_incorporation}"
                )

        # generate response
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)
        return AssistantResponse(
            response=response[0][0],
            contexts=contexts,
            metadata={"prompt": prompt},
        )
