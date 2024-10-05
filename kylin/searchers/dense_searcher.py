import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field

from kylin.prompt import ChatTurn, ChatPrompt
from kylin.retriever import DenseRetriever, DenseRetrieverConfig, RetrievedContext
from kylin.utils import Choices

from .searcher import AgentSearcher, AgentSearcherConfig, Searchers, SearchHistory


logger = logging.getLogger("DenseSearcher")


@dataclass
class DenseSearcherConfig(AgentSearcherConfig):
    retriever_config: DenseRetrieverConfig = field(default_factory=DenseRetrieverConfig)
    rewrite_query: Choices(["never", "pseudo", "adaptive"]) = "never"  # type: ignore
    max_rewrite_depth: int = 3
    retriever_top_k: int = 10
    disable_cache: bool = False


@Searchers("dense", config_class=DenseSearcherConfig)
class DenseSearcher(AgentSearcher):
    is_hybrid = False

    def __init__(self, cfg: DenseSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Dense Searcher
        self.rewrite = cfg.rewrite_query
        self.rewrite_depth = cfg.max_rewrite_depth
        self.retriever_top_k = cfg.retriever_top_k
        self.disable_cache = cfg.disable_cache

        # load Dense Retrieve
        self.retriever = DenseRetriever(cfg.retriever_config)

        # load prompts
        self.rewrite_with_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "rewrite_by_answer_with_context_prompt.json",
            )
        )
        self.rewrite_wo_ctx_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "rewrite_by_answer_without_context_prompt.json",
            )
        )
        self.verify_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
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
            ctxs = self.retriever.search(
                query=[query_to_search],
                top_k=self.retriever_top_k,
                disable_cache=self.disable_cache,
            )[0]
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
            for n, ctx in enumerate(contexts):
                user_prompt += f"Context {n}: {ctx.text}\n\n"
            user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return f"{question} {query}"

    def verify_contexts(
        self,
        contexts: list[RetrievedContext],
        question: str,
    ) -> bool:
        prompt = deepcopy(self.verify_prompt)
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx.text}\n\n"
        user_prompt += f"Question: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return "yes" in response.lower()

    def close(self) -> None:
        self.retriever.close()
        return
