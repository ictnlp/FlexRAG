import logging
import os
import re
from copy import deepcopy
from dataclasses import dataclass, field

from kylin.prompt import ChatTurn, ChatPrompt
from kylin.retriever import (
    ElasticRetriever,
    ElasticRetrieverConfig,
    RetrievedContext,
    TypesenseRetriever,
    TypesenseRetrieverConfig,
)
from kylin.utils import Choices
from kylin.retriever.keyword import Keywords, Keyword

from .searcher import AgentSearcher, AgentSearcherConfig, Searchers, SearchHistory


logger = logging.getLogger("KeywordSearcher")


@dataclass
class KeywordSearcherConfig(AgentSearcherConfig):
    retriever_type: Choices(["elastic", "typesense"]) = "elastic"  # type: ignore
    elastic_config: ElasticRetrieverConfig = field(default_factory=ElasticRetrieverConfig)  # fmt: skip
    typesense_config: TypesenseRetrieverConfig = field(default_factory=TypesenseRetrieverConfig)  # fmt: skip
    rewrite_query: Choices(["always", "never", "adaptive"]) = "never"  # type: ignore
    feedback_depth: int = 1


@Searchers("keyword", config_class=KeywordSearcherConfig)
class KeywordSearcher(AgentSearcher):
    is_hybrid = False

    def __init__(self, cfg: KeywordSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Keyword Searcher
        self.rewrite = cfg.rewrite_query
        self.feedback_depth = cfg.feedback_depth
        self.retriever_top_k = cfg.retriever_top_k
        self.disable_cache = cfg.disable_cache

        # load Retriever
        match cfg.retriever_type:
            case "elastic":
                self.retriever = ElasticRetriever(cfg.elastic_config)
            case "typesense":
                self.retriever = TypesenseRetriever(cfg.typesense_config)
            case _:
                raise ValueError(f"Retriever {cfg.retriever_type} not supported")

        # load prompts
        self.rewrite_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "keyword_rewrite_prompt.json",
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
                    "keyword_refine_extend_prompt.json",
                )
            ),
            "filter": ChatPrompt.from_json(
                os.path.join(
                    os.path.dirname(__file__),
                    "searcher_prompts",
                    "keyword_refine_filter_prompt.json",
                )
            ),
            "emphasize": ChatPrompt.from_json(
                os.path.join(
                    os.path.dirname(__file__),
                    "searcher_prompts",
                    "keyword_refine_emphasize_prompt.json",
                )
            ),
        }
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        history: list[SearchHistory] = []

        # rewrite the query into keywords
        match self.rewrite:
            case "always":
                query_to_search = self.rewrite_query(question)
            case "never":
                ctxs = self.retriever.search(
                    query=[question],
                    top_k=self.retriever_top_k,
                    disable_cache=self.disable_cache,
                    search_method="full_text",
                )[0]
                history.append(SearchHistory(query=question, contexts=ctxs))
                return ctxs, history
            case "adaptive":
                ctxs = self.retriever.search(
                    query=[question],
                    top_k=self.retriever_top_k,
                    disable_cache=self.disable_cache,
                    search_method="full_text",
                )[0]
                verification = self.verify_contexts(ctxs, question)
                history.append(SearchHistory(query=question, contexts=ctxs))
                if verification:
                    return ctxs, history
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
                search_method="keyword",
            )[0]
            history.append(SearchHistory(query=str(query_to_search), contexts=ctxs))

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
        return ctxs, history

    def refine_query(
        self,
        contexts: list[RetrievedContext],
        base_query: str,
        current_query: Keywords,
    ) -> list[Keywords]:
        refined_queries = []
        prompts = []
        for prompt_type in self.refine_prompts:
            # prepare prompt
            prompt = deepcopy(self.refine_prompts[prompt_type])
            ctx_str = ""
            for n, ctx in enumerate(contexts):
                ctx_str += f"Context {n}: {ctx.text}\n\n"
            q_str = "Current keywords:\n"
            must_contains = []
            must_not_contains = []
            for kw in current_query:
                if kw.must:
                    must_contains.append(kw.keyword)
                elif kw.must_not:
                    must_not_contains.append(kw.keyword)
                else:
                    q_str += f"Keyword: {kw.keyword}\nWeight: {kw.weight}\n\n"
            q_str += f"Must contained keywords:\n"
            for kw in must_contains:
                q_str += f"{kw}\n"
            q_str += f"\nMust not contained keywords:\n"
            for kw in must_not_contains:
                q_str += f"{kw}\n"
            q_str += "\n"
            prompt.history[-1].content = (
                f"{ctx_str}"
                f"{prompt.history[-1].content}\n\n"
                f"{q_str}\n\n"
                f"The information you are looking for: {base_query}"
            )
            prompts.append(prompt)
        responses = self.agent.chat(prompts, generation_config=self.gen_cfg)

        for prompt_type, response in zip(self.refine_prompts, responses):
            # prepare re patterns
            response = response[0]
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

    def rewrite_query(self, query: str) -> Keywords:
        # Rewrite the query to be more informative
        prompt = deepcopy(self.rewrite_prompt)
        prompt.update(ChatTurn(role="user", content=query))
        response = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        # Parse the response into keywords
        keywords = response.split("\n")
        keywords = [Keyword(keyword=kw) for kw in keywords if kw]
        return keywords

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
