import os
from copy import deepcopy
from dataclasses import dataclass

from librarian.assistant import ASSISTANTS, SearchHistory
from librarian.prompt import ChatPrompt, ChatTurn
from librarian.retriever import (
    DuckDuckGoRetriever,
    DuckDuckGoRetrieverConfig,
    RetrievedContext,
)
from librarian.utils import LOGGER_MANAGER

from .searcher import AgentSearcher, AgentSearcherConfig

logger = LOGGER_MANAGER.get_logger("librarian.examples.level_rag")


@dataclass
class WebSearcherConfig(AgentSearcherConfig, DuckDuckGoRetrieverConfig):
    rewrite_query: bool = False


@ASSISTANTS("levelrag_web", config_class=WebSearcherConfig)
class WebSearcher(AgentSearcher):
    def __init__(self, cfg: WebSearcherConfig) -> None:
        super().__init__(cfg)
        # setup web assistant
        self.rewrite = cfg.rewrite_query

        # load Web Retrieve
        self.retriever = DuckDuckGoRetriever(cfg)
        assert all(
            field in self.retriever.fields for field in self.used_fields
        ), f"retriever does not have field {self.used_fields}"

        # load rewrite prompt
        self.rewrite_prompt = ChatPrompt.from_json(
            os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "web_rewrite_prompt.json",
            )
        )
        return

    def search(
        self, question: str
    ) -> tuple[list[RetrievedContext], list[SearchHistory]]:
        # initialize search stack
        if self.rewrite:
            query_to_search = self.rewrite_query(question)
        else:
            query_to_search = question
        ctxs = self.retriever.search(query=[query_to_search])[0]
        return ctxs, [SearchHistory(query=question, contexts=ctxs)]

    def rewrite_query(self, info: str) -> str:
        # Rewrite the query to be more informative
        user_prompt = f"Query: {info}"
        prompt = deepcopy(self.rewrite_prompt)
        prompt.update(ChatTurn(role="user", content=user_prompt))
        query = self.agent.chat([prompt], generation_config=self.cfg)[0][0]
        return query
