from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from flexrag.common_dataclass import RetrievedContext
from flexrag.context_refine import BasicPacker, BasicPackerConfig
from flexrag.models import GENERATORS, GenerationConfig, GeneratorConfig
from flexrag.prompt import ChatPrompt, ChatTurn
from flexrag.ranker import RANKERS, RankerConfig
from flexrag.retriever import RETRIEVERS, RetrieverConfig
from flexrag.utils import LOGGER_MANAGER, Choices

from .assistant import ASSISTANTS, PREDEFINED_PROMPTS, AssistantBase, SearchHistory

logger = LOGGER_MANAGER.get_logger("flexrag.assistant.modular")


@dataclass
class ModularAssistantConfig(
    GeneratorConfig, GenerationConfig, RetrieverConfig, RankerConfig, BasicPackerConfig
):
    """The configuration for the modular assistant.

    :param response_type: The type of response to generate.
        Defaults to "short". Available options are: "short", "long", "original", "custom".
    :type response_type: str, optional
    :param prompt_with_context_path: The path to the prompt file for response with context. Defaults to None.
    :type prompt_with_context_path: str, optional
    :param prompt_without_context_path: The path to the prompt file for response without context. Defaults to None.
    :type prompt_without_context_path: str, optional
    :param used_fields: The fields to use in the context. Defaults to [].
    :type used_fields: list[str], optional
    """

    response_type: Choices(["short", "long", "original", "custom"]) = "short"  # type: ignore
    prompt_with_context_path: Optional[str] = None
    prompt_without_context_path: Optional[str] = None
    used_fields: list[str] = field(default_factory=list)


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

        # load generator
        self.generator = GENERATORS.load(cfg)
        assert self.generator is not None, "Generator is not loaded."

        # load retriever
        self.retriever = RETRIEVERS.load(cfg)

        # load ranker
        self.reranker = RANKERS.load(cfg)

        # load packer
        self.context_packer = BasicPacker(cfg)

        # load prompts
        match cfg.response_type:
            case "short":
                self.prompt_with_ctx = PREDEFINED_PROMPTS["shortform_with_context"]
                self.prompt_wo_ctx = PREDEFINED_PROMPTS["shortform_without_context"]
            case "long":
                self.prompt_with_ctx = PREDEFINED_PROMPTS["longform_with_context"]
                self.prompt_wo_ctx = PREDEFINED_PROMPTS["longform_without_context"]
            case "original":
                self.prompt_with_ctx = ChatPrompt()
                self.prompt_wo_ctx = ChatPrompt()
            case "custom":
                self.prompt_with_ctx = ChatPrompt.from_json(
                    cfg.prompt_with_context_path
                )
                self.prompt_wo_ctx = ChatPrompt.from_json(
                    cfg.prompt_without_context_path
                )
            case _:
                raise ValueError(f"Invalid response type: {cfg.response_type}")
        return

    def answer(
        self, question: str
    ) -> tuple[str, list[RetrievedContext], dict[str, Any]]:
        ctxs, history = self.search(question)
        response, prompt = self.answer_with_contexts(question, ctxs)
        return response, ctxs, {"prompt": prompt, "search_histories": history}

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        if self.retriever is None:
            return [], []
        # searching for contexts
        search_histories = []
        ctxs = self.retriever.search(query=[question])[0]
        search_histories.append(SearchHistory(query=question, contexts=ctxs))

        # reranking
        if self.reranker is not None:
            results = self.reranker.rank(question, ctxs)
            ctxs = results.candidates
            search_histories.append(SearchHistory(query=question, contexts=ctxs))

        # packing
        if len(ctxs) > 1:
            ctxs = self.context_packer.refine(ctxs)
            search_histories.append(SearchHistory(query=question, contexts=ctxs))

        return ctxs, search_histories

    def answer_with_contexts(
        self, question: str, contexts: list[RetrievedContext] = []
    ) -> tuple[str, ChatPrompt]:
        # prepare system prompts
        if len(contexts) > 0:
            prompt = deepcopy(self.prompt_with_ctx)
        else:
            prompt = deepcopy(self.prompt_wo_ctx)

        # prepare user prompt
        usr_prompt = ""
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
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=usr_prompt))

        # generate response
        response = self.generator.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return response, prompt
