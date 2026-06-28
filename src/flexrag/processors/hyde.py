from typing import Annotated

from flexrag.common import Choices, configure, trace
from flexrag.models.generators import GeneratorProtocol


class HydeRewriter:
    """Rewrite queries with HyDE prompts.

    HyDE is a lightweight query rewriting helper for the classic
    Hypothetical Document Embeddings method. It only owns prompt construction
    and generator calls; retriever lifecycle, query preprocessing, caching, and
    search are handled by the caller.

    The original paper is available at https://aclanthology.org/2023.acl-long.99/.
    """

    Prompts = {
        "WEB_SEARCH": "Please write a passage to answer the question.\nQuestion: {}\nPassage:",
        "SCIFACT": "Please write a scientific paper passage to support/refute the claim.\nClaim: {}\nPassage:",
        "ARGUANA": "Please write a counter argument for the passage.\nPassage: {}\nCounter Argument:",
        "TREC_COVID": "Please write a scientific paper passage to answer the question.\nQuestion: {}\nPassage:",
        "FIQA": "Please write a financial article passage to answer the question.\nQuestion: {}\nPassage:",
        "DBPEDIA_ENTITY": "Please write a passage to answer the question.\nQuestion: {}\nPassage:",
        "TREC_NEWS": "Please write a news passage about the topic.\nTopic: {}\nPassage:",
        "MR_TYDI": "Please write a passage in {} to answer the question in detail.\nQuestion: {}\nPassage:",
    }

    def __init__(self, cfg: "HydeConfig", generator: GeneratorProtocol) -> None:
        """Initialize the HyDE rewriter.

        :param cfg: HyDE prompt configuration.
        :param generator: Generator used to produce hypothetical passages.
        """
        self.cfg = cfg
        self.generator = generator
        return

    def _format_prompt(self, query: str) -> str:
        if self.cfg.task == "MR_TYDI":
            return self.Prompts[self.cfg.task].format(self.cfg.language, query)
        return self.Prompts[self.cfg.task].format(query)

    @trace("processor.hyde.rewrite")
    def rewrite(self, queries: list[str] | str) -> list[str]:
        """Rewrite one query or a batch of queries.

        :param queries: Query or queries to rewrite.
        :return: Hypothetical passages generated from the configured HyDE
            prompt, one per input query.
        """
        if isinstance(queries, str):
            queries = [queries]
        prompts = [self._format_prompt(q) for q in queries]
        return [response[0] for response in self.generator.generate(prompts)]


@configure
class HydeConfig:
    """Configuration for :class:`HydeRewriter`.

    :param task: Prompt template used to rewrite the query. Defaults to
        ``"WEB_SEARCH"``. Available options are ``"WEB_SEARCH"``,
        ``"SCIFACT"``, ``"ARGUANA"``, ``"TREC_COVID"``, ``"FIQA"``,
        ``"DBPEDIA_ENTITY"``, ``"TREC_NEWS"``, and ``"MR_TYDI"``.
    :param language: Target language used by the ``"MR_TYDI"`` prompt.
        Defaults to ``"en"``.
    """

    task: Annotated[str, Choices(*HydeRewriter.Prompts.keys())] = "WEB_SEARCH"
    language: str = "en"


__all__ = [
    "HydeConfig",
    "HydeRewriter",
]
