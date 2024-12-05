import os
from copy import deepcopy
from dataclasses import dataclass

from librarian.assistant import ASSISTANTS, SearchHistory
from librarian.prompt import ChatPrompt, ChatTurn
from librarian.retriever import DenseRetriever, DenseRetrieverConfig, RetrievedContext
from librarian.utils import LOGGER_MANAGER, Choices

from .searcher import AgentSearcher, AgentSearcherConfig

logger = LOGGER_MANAGER.get_logger("librarian.examples.level_rag")


@dataclass
class DenseSearcherConfig(AgentSearcherConfig, DenseRetrieverConfig):
    rewrite_query: Choices(["never", "hyde", "itrg"]) = "never"  # type: ignore
    max_rewrite_times: int = 3


@ASSISTANTS("levelrag_dense", config_class=DenseSearcherConfig)
class DenseSearcher(AgentSearcher):
    def __init__(self, cfg: DenseSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Dense Assistant
        self.rewrite = cfg.rewrite_query
        self.rewrite_depth = cfg.max_rewrite_times

        # load Dense Retrieve
        self.retriever = DenseRetriever(cfg)
        assert all(
            field in self.retriever.fields for field in self.used_fields
        ), f"retriever does not have field {self.used_fields}"

        # load rewrite prompts
        self.rewrite_with_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "rewrite_by_answer_with_context_prompt.json",
            )
        )
        self.rewrite_wo_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "rewrite_by_answer_without_context_prompt.json",
            )
        )
        self.verify_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "verify_prompt.json",
            )
        )
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        # rewrite the query
        if self.rewrite == "pseudo":
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question

        # begin adaptive search
        ctxs = []
        history: list[SearchHistory] = []
        verification = False
        rewrite_depth = 0
        while (not verification) and (rewrite_depth < self.rewrite_depth):
            rewrite_depth += 1

            # search
            ctxs = self.retriever.search(query=[query_to_search])[0]
            history.append(SearchHistory(query=query_to_search, contexts=ctxs))

            # verify the contexts
            if self.rewrite == "adaptive":
                verification = self.verify_contexts(ctxs, question)
            else:
                verification = True

            # adaptive rewrite
            if (not verification) and (rewrite_depth < self.rewrite_depth):
                if rewrite_depth == 1:
                    query_to_search = self.rewrite_query(question)
                else:
                    query_to_search = self.rewrite_query(question, ctxs)
        return ctxs, history

    def rewrite_query(
        self, question: str, contexts: list[RetrievedContext] = []
    ) -> str:
        # Rewrite the query to be more informative
        if len(contexts) == 0:
            prompt = deepcopy(self.rewrite_wo_ctx_prompt)
            user_prompt = f"Question: {question}"
        else:
            prompt = deepcopy(self.rewrite_with_ctx_prompt)
            user_prompt = ""
            for n, context in enumerate(contexts):
                if len(self.used_fields) == 1:
                    ctx = context.data[self.used_fields[0]]
                else:
                    ctx = ""
                    for field_name in self.used_fields:
                        ctx += f"{field_name}: {context.data[field_name]}\n"
                user_prompt += f"Context {n + 1}: {ctx}\n\n"
            user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.cfg)[0][0]
        return f"{question} {query}"

    def verify_contexts(
        self,
        contexts: list[RetrievedContext],
        question: str,
    ) -> bool:
        prompt = deepcopy(self.verify_prompt)
        user_prompt = ""
        for n, context in enumerate(contexts):
            if len(self.used_fields) == 1:
                ctx = context.data[self.used_fields[0]]
            else:
                ctx = ""
                for field_name in self.used_fields:
                    ctx += f"{field_name}: {context.data[field_name]}\n"
            user_prompt += f"Context {n + 1}: {ctx}\n\n"
        user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        response = self.agent.chat([prompt], generation_config=self.cfg)[0][0]
        return "yes" in response.lower()
