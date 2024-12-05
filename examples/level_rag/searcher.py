from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass

from omegaconf import MISSING

from librarian.assistant import PREDEFINED_PROMPTS, AssistantBase, SearchHistory
from librarian.models import GENERATORS, GenerationConfig
from librarian.prompt import ChatPrompt, ChatTurn
from librarian.retriever import RetrievedContext
from librarian.utils import Choices, Register, LoggerManager

logger = LoggerManager.get_logger(__name__)


Searchers = Register("searchers")
GeneratorConfig = GENERATORS.make_config()


@dataclass
class AgentSearcherConfig(GenerationConfig, GeneratorConfig):
    used_fields: list[str] = MISSING
    response_type: Choices(["short", "long", "original"]) = "short"  # type: ignore


class AgentSearcher(AssistantBase):
    def __init__(self, cfg: AgentSearcherConfig) -> None:
        super().__init__()
        # set basic args
        self.used_fields = cfg.used_fields

        # load generator
        self.agent = GENERATORS.load(cfg)
        self.cfg = cfg
        if self.cfg.sample_num > 1:
            logger.warning("sample_num > 1 is not supported by the AgentSearcher")
            logger.warning("Setting the sample_num to 1")
            self.cfg.sample_num = 1

        # load generation prompts
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
            case _:
                raise ValueError(f"Invalid response type: {cfg.response_type}")
        return

    @abstractmethod
    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        return

    def answer(
        self, question: str
    ) -> tuple[str, list[RetrievedContext] | None, dict | None]:
        ctxs, history = self.search(question)
        response, prompt = self.answer_with_contexts(question, ctxs)
        return response, ctxs, {"prompt": prompt, "search_histories": history}

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
            used_fields = set(self.used_fields) & set(context.data.keys())
            if len(used_fields) == 1:
                ctx = context.data[used_fields.pop()]
            else:
                ctx = ""
                for field_name in used_fields:
                    ctx += f"{field_name}: {context.data[field_name]}\n"
            usr_prompt += f"Context {n + 1}: {ctx}\n\n"
        usr_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=usr_prompt))

        # generate response
        response = self.agent.chat([prompt], generation_config=self.cfg)[0][0]
        return response, prompt
