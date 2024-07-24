import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field

from kylin.prompt import ChatTurn, ChatPrompt
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
    retriever_top_k: int = 10
    disable_cache: bool = False
# fmt: on


class WebSearcher(BaseSearcher):
    def __init__(self, cfg: WebSearcherConfig) -> None:
        super().__init__(cfg)
        # setup Web Searcher
        self.rewrite = cfg.rewrite_query
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

        # load prompt
        self.rewrite_prompt = ChatPrompt.from_file(
            os.path.join(
                os.path.dirname(__file__),
                "searcher_prompts",
                "web_rewrite_prompt.json",
            )
        )
        return

    def search(
        self, question: str
    ) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
        # initialize search stack
        if self.rewrite:
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question
        ctxs = self.retriever.search(
            query=[query_to_search],
            top_k=self.retriever_top_k,
            disable_cache=self.disable_cache,
        )[0]
        return ctxs, []

    def rewrite_query(self, info: str) -> str:
        # Rewrite the query to be more informative
        user_prompt = f"Query: {info}"
        prompt = deepcopy(self.rewrite_prompt)
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.gen_cfg)[0][0]
        return query

    def close(self) -> None:
        self.retriever.close()
        return
