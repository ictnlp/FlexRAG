from dataclasses import field
from typing import Any, Optional

from flexrag.common import LOGGER_MANAGER, configure, data
from flexrag.common.dataclasses import ChatMessages, RetrievedContext
from flexrag.models.generators import GenerationConfig, GeneratorProtocol
from flexrag.processors.rankers.ranker_base import RankerBase
from flexrag.processors.refiners.refiner_base import RefinerBase
from flexrag.retrievers.retriever_base import RetrieverBase

from .assistant_base import ASSISTANTS, AssistantBase, AssistantResponse

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
class ModularAssistantConfig(GenerationConfig):
    """The configuration for the modular assistant.

    :param used_fields: The fields to use in the context. Defaults to [].
    """

    used_fields: list[str] = field(default_factory=list)


@ASSISTANTS("modular", config_class=ModularAssistantConfig)
class ModularAssistant(AssistantBase):
    """The modular RAG assistant that supports retrieval, reranking, and generation."""

    def __init__(
        self,
        cfg: ModularAssistantConfig,
        generator: GeneratorProtocol,
        retriever: RetrieverBase | None = None,
        reranker: RankerBase | None = None,
        refiners: list[RefinerBase] | None = None,
    ):
        # set basic args
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1
        self.used_fields = cfg.used_fields

        # attach injected resources
        self.generator = generator
        self.retriever = retriever
        self.reranker = reranker
        self.refiners = list(refiners or [])
        return

    def answer(
        self,
        messages: ChatMessages | list[dict],
        additional_sessions: list[ChatMessages] | None = None,
    ) -> AssistantResponse:
        if isinstance(messages, list):
            messages = ChatMessages.from_list(messages)
        ctxs, search_history = [], []
        if self.retriever is not None:
            ctxs, search_history = self.search(messages[-1].content)
        response = self.generate_response(messages, ctxs)
        if response.metadata is None:
            response.metadata = {}
        response.metadata["search_histories"] = search_history
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
        for refiner in self.refiners:
            ctxs = refiner.refine(ctxs)
            search_histories.append(
                SearchResult(query=f"refine: {query}", contexts=ctxs)
            )

        return ctxs, search_histories

    def generate_response(
        self,
        messages: ChatMessages | list[dict],
        contexts: list[RetrievedContext] | None = None,
    ) -> AssistantResponse:
        # convert messages to ChatMessages if it's a list of dicts
        if isinstance(messages, list):
            messages = ChatMessages.from_list(messages)
        contexts = contexts or []

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
        prompt[-1].content = (
            f"Here are some context documents that may be relevant to this conversation:\n\n"
            f"{context_str}\n"
            f"{prompt[-1].content}"
        )

        # generate response
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)
        return AssistantResponse(
            response=response[0][0],
            contexts=contexts,
            metadata={"prompt": prompt},
        )
