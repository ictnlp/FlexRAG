import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field

from kylin.prompt import ChatTurn, ChatPrompt
from kylin.retriever import ElasticRetriever, ElasticRetrieverConfig, RetrievedContext
from kylin.utils import Choices

from .searcher import BaseSearcher, BaseSearcherConfig, Searchers


logger = logging.getLogger("LuceneSearcher")


@dataclass
class LuceneSearcherConfig(BaseSearcherConfig):
    retriever_config: ElasticRetrieverConfig = field(
        default_factory=ElasticRetrieverConfig
    )
    rewrite_query: Choices(["always", "never", "adaptive"]) = "never"  # type: ignore
    feedback_depth: int = 1
    retriever_top_k: int = 10
    disable_cache: bool = False


@Searchers("lucene", config_class=LuceneSearcherConfig)
class LuceneSearcher(BaseSearcher):
    def __init__(self, cfg: LuceneSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Lucene Searcher
        self.rewrite = cfg.rewrite_query
        self.feedback_depth = cfg.feedback_depth
        self.retriever_top_k = cfg.retriever_top_k
        self.disable_cache = cfg.disable_cache

        # load Lucene Retriever
        self.retriever = ElasticRetriever(cfg.retriever_config)

        # load prompts
        self.rewrite_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "lucene_rewrite_prompt.json",
            )
        )
        self.verify_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "verify_prompt.json",
            )
        )
        self.refine_prompts = {
            "extend": ChatPrompt.from_json(
                os.path.join(
                    os.path.dirname(__file__),
                    "searcher_prompts",
                    "lucene_refine_extend_prompt.json",
                )
            ),
            "filter": ChatPrompt.from_json(
                os.path.join(
                    os.path.dirname(__file__),
                    "searcher_prompts",
                    "lucene_refine_filter_prompt.json",
                )
            ),
            "emphasize": ChatPrompt.from_json(
                os.path.join(
                    os.path.dirname(__file__),
                    "searcher_prompts",
                    "lucene_refine_emphasize_prompt.json",
                )
            ),
        }
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[dict[str, object]]]:
        retrieval_history = []

        # rewrite the query
        match self.rewrite:
            case "always":
                query_to_search = self.rewrite_query(question)
            case "never":
                query_to_search = question
            case "adaptive":
                ctxs = self.retriever.search(
                    query=[question],
                    top_k=self.retriever_top_k,
                    disable_cache=self.disable_cache,
                )[0]
                verification = self.verify_contexts(ctxs, question)
                retrieval_history.append(
                    {
                        "query": question,
                        "ctxs": ctxs,
                    }
                )
                if verification:
                    return ctxs, retrieval_history
                query_to_search = self.rewrite_query(question)

        # begin BFS search
        search_stack = [(query_to_search, 1)]
        total_depth = self.feedback_depth + 1
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
        contexts: list[RetrievedContext],
        base_query: str,
        current_query: str,
    ) -> list[str]:
        refined_queries = []
        for prompt_type in self.refine_prompts:
            # prepare prompt
            prompt = deepcopy(self.refine_prompts[prompt_type])
            ctx_str = ""
            for n, ctx in enumerate(contexts):
                ctx_str += f"Context {n}: {ctx.text}\n\n"
            prompt.history[-1].content = (
                f"{ctx_str}{prompt.history[-1].content}\n\n"
                f"Current query: {current_query}\n\n"
                f"The information you are looking for: {base_query}"
            )
            response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]

            # prepare re patterns
            response_ = re.escape(response)
            pattern = f'("{response_}"(\^\d)?)|({response_})'

            # append refined query
            if prompt_type == "extend":
                refined_queries.append(f"{current_query} {response}")
            elif prompt_type == "filter":
                if re.search(pattern, current_query):
                    refined_queries.append(re.sub(pattern, "", current_query))
                else:
                    refined_queries.append(f'{current_query} -"{response}"')
            elif prompt_type == "emphasize":
                if re.search(pattern, current_query):
                    try:
                        current_weight = re.search(pattern, current_query).group(2)
                        current_weight = int(current_weight[1:])
                    except:
                        current_weight = 1
                    repl = re.escape(f'"{response}"^{current_weight + 1}')
                    new_query = re.sub(pattern, repl, current_query)
                    refined_queries.append(new_query)
                else:
                    refined_queries.append(f'"{response}" {current_query}')
        return refined_queries

    def rewrite_query(self, info: str) -> str:
        # Rewrite the query to be more informative
        prompt = deepcopy(self.rewrite_prompt)
        prompt.update(ChatTurn(role="user", content=info))
        query = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return query

    def verify_contexts(
        self,
        contexts: list[RetrievedContext],
        question: str,
    ) -> bool:
        prompt = deepcopy(self.verify_prompt)
        user_prompt = ""
        for n, ctx in enumerate(contexts):
            user_prompt += f"Context {n}: {ctx.text}\n\n"
        user_prompt += f"Topic: {question}"
        prompt.update(ChatTurn(role="user", content=user_prompt))
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return "yes" in response.lower()

    def close(self) -> None:
        self.retriever.close()
        return
