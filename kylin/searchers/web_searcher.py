import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field

from kylin.kylin_prompts import *
from kylin.retriever import (
    DuckDuckGoRetriever,
    DuckDuckGoRetrieverConfig,
    BingRetriever,
    BingRetrieverConfig,
)
from kylin.utils import Choices

from .searcher import BaseSearcher, SearcherConfig


logger = logging.getLogger("WebSearcher")


# fmt: off
@dataclass
class WebSearcherConfig(SearcherConfig):
    ddg_config: DuckDuckGoRetrieverConfig = field(default_factory=DuckDuckGoRetrieverConfig)
    bing_config: BingRetrieverConfig = field(default_factory=BingRetrieverConfig)
    retriever_type: Choices(["ddg", "bing"]) = "ddg"  # type: ignore
    rewrite_query: bool = False
    summarize_context: bool = False
    feedback_depth: int = 1
    retriever_top_k: int = 10
    disable_cache: bool = False
# fmt: on


class WebSearcher(BaseSearcher):
    def __init__(self, cfg: WebSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Web Searcher
        self.rewrite = cfg.rewrite_query
        self.feedback_depth = cfg.feedback_depth
        self.summary_context = cfg.summarize_context
        self.retriever_top_k = cfg.retriever_top_k
        self.disable_cache = cfg.disable_cache

        # load Web Retrieve
        match cfg.retriever_type:
            case "ddg":
                self.retriever = DuckDuckGoRetriever(cfg.ddg_config)
            case "bing":
                self.retriever = BingRetriever(cfg.bing_config)
            case _:
                raise ValueError(f"Invalid retriever type: {cfg.retriever_type}")
        return

    def search(
        self, question: str
    ) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
        retrieval_history = []

        # initialize search stack
        if self.rewrite:
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question
        search_stack = [(query_to_search, 1)]
        total_depth = self.feedback_depth + 1

        # begin BFS search
        while len(search_stack) > 0:
            # search
            query_to_search, depth = search_stack.pop(0)
            ctxs = self.retriever.search(
                query=[query_to_search],
                top_k=self.retriever_top_k,
                disable_cache=self.disable_cache,
            )[0]
            retrieval_history.append(
                {
                    "query": query_to_search,
                    "ctxs": ctxs,
                }
            )

            # verify contexts
            if total_depth > 1:
                verification = self.verify_contexts(ctxs, question)
            else:
                verification = True
            if verification:
                break

            # if depth is already at the maximum, stop expanding
            if depth >= total_depth:
                continue

            # expand the search stack
            refined = self.refine_query(
                contexts=ctxs,
                base_query=question,
                current_query=query_to_search,
            )
            search_stack.extend([(rq, depth + 1) for rq in refined])
        return ctxs, retrieval_history

    def refine_query(
        self,
        contexts: list[str],
        base_query: str,
        current_query: str,
    ):
        refined_queries = []
        for prompt_type in refine_prompt["bm25"]:
            prompt = deepcopy(refine_prompt["bm25"][prompt_type])
            ctx_str = ""
            for n, ctx in enumerate(contexts):
                ctx_str += f"Context {n}: {ctx['text']}\n\n"
            prompt[-1]["content"] = ctx_str + prompt[-1]["content"]
            prompt[-1]["content"] += f"\n\nCurrent query: {current_query}"
            prompt[-1][
                "content"
            ] += f"\n\nThe information you are looking for: {base_query}"
            response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0]
            if prompt_type == "extend":
                refined_queries.append(f"{current_query} {response}")
            elif prompt_type == "filter":
                refined_queries.append(f"{current_query} -{response}")
            elif prompt_type == "emphasize":
                if re.search(f'"{response}"', current_query) is None:
                    new_query = re.sub(
                        f'"{response}"', f'"{response}"^2', current_query
                    )
                    refined_queries.append(new_query)
                refined_queries
            return refined_queries

    def rewrite_query(self, info: str) -> str:
        # Rewrite the query to be more informative
        user_prompt = f"Query: {info}"
        prompt = deepcopy(rewrite_prompts["web"])
        prompt.append({"role": "user", "content": user_prompt})
        query_to_search = self.agent.chat([prompt], generation_config=self.gen_cfg)[0]
        return query_to_search

    def verify_contexts(
        self,
        contexts: list[dict[str, str]],
        question: str,
    ) -> bool:
        prompt = deepcopy(verify_prompt)
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx['text']}\n\n"
        user_prompt += f"Topic: {question}"
        prompt.append({"role": "user", "content": user_prompt})
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0]
        return "yes" in response.lower()

    def close(self) -> None:
        self.retriever.close()
        return
