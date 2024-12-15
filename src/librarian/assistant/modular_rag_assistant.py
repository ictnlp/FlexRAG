from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

from librarian.context_refine import BasicPacker, BasicPackerConfig
from librarian.models import GENERATORS, GenerationConfig
from librarian.prompt import ChatPrompt, ChatTurn
from librarian.ranker import RANKERS
from librarian.retriever import RETRIEVERS, RetrievedContext
from librarian.utils import Choices, LOGGER_MANAGER

from .assistant import ASSISTANTS, AssistantBase, SearchHistory, PREDEFINED_PROMPTS

logger = LOGGER_MANAGER.get_logger("librarian.assistant.modular")


GeneratorConfig = GENERATORS.make_config()
RetrieverConfig = RETRIEVERS.make_config(default=None)
RankerConfig = RANKERS.make_config(default=None)


@dataclass
class ModularAssistantConfig(
    GeneratorConfig, GenerationConfig, RetrieverConfig, RankerConfig, BasicPackerConfig
):
    response_type: Choices(["short", "long", "original", "custom"]) = "short"  # type: ignore
    prompt_with_context_path: Optional[str] = None
    prompt_without_context_path: Optional[str] = None
    used_fields: list[str] = field(default_factory=list)


@ASSISTANTS("modular", config_class=ModularAssistantConfig)
class ModularAssistant(AssistantBase):
    def __init__(self, cfg: ModularAssistantConfig):
        # set basic args
        self.gen_cfg = cfg
        if self.gen_cfg.sample_num > 1:
            logger.warning("Sample num > 1 is not supported for Assistant")
            self.gen_cfg.sample_num = 1
        self.used_fields = cfg.used_fields

        # load generator
        self.generator = GENERATORS.load(cfg)

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
        """Answer the given question.

        Args:
            question (str): The question to answer.

        Returns:
            response (str): The response to the question.
            contexts (list[RetrievedContext]): The contexts used to answer the question.
            metadata (dict): The chatprompt and the context processing history used by the assistant.
        """
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
