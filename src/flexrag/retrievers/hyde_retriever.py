from collections.abc import Iterable
from typing import Annotated, Any

from flexrag.common import Choices, configure, trace
from flexrag.common.dataclasses import RetrievedContext
from flexrag.models.generators import GeneratorProtocol

from .retriever_base import (
    DEFAULT_TOP_K,
    RETRIEVERS,
    RetrieverBase,
    RetrieverBaseConfig,
)


class HydeRewriter:
    """Rewrite search queries with HyDE prompts before retrieval.

    The rewriter owns only the prompt construction and generator call. It does
    not search indexes, manage retriever state, or apply query preprocessing.
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

    def __init__(self, generator: GeneratorProtocol, task: str, language: str = "en"):
        self.task = task
        self.language = language
        self.generator = generator
        return

    def _format_prompt(self, query: str) -> str:
        if self.task == "MR_TYDI":
            return self.Prompts[self.task].format(self.language, query)
        return self.Prompts[self.task].format(query)

    @trace("retriever.hyde_retriever.rewrite")
    def rewrite(self, queries: list[str] | str) -> list[str]:
        """Rewrite one query or a batch of queries.

        :param queries: Query or queries to rewrite.
        :return: Hypothetical passages generated from the configured HyDE
            prompt.
        """
        if isinstance(queries, str):
            queries = [queries]
        prompts = [self._format_prompt(q) for q in queries]
        new_queries = [q[0] for q in self.generator.generate(prompts)]
        return new_queries


@configure
class HydeRetrieverConfig(RetrieverBaseConfig):
    """Configuration class for HydeRetriever.

    :param task: Task for rewriting the query. Default: "WEB_SEARCH".
        Available options: "WEB_SEARCH", "SCIFACT", "ARGUANA", "TREC_COVID", "FIQA", "DBPEDIA_ENTITY", "TREC_NEWS", "MR_TYDI".
    :param language: Language for rewriting. Default: "en".
    :param batch_size: Batch size used by the wrapper when rewriting queries.
        The wrapped retriever owns retrieval batching.
    :param query_preprocess_pipeline: Ignored by this wrapper. Configure query
        preprocessing on the wrapped retriever instead.
    """

    task: Annotated[str, Choices(*HydeRewriter.Prompts.keys())] = "WEB_SEARCH"
    language: str = "en"


@RETRIEVERS("hyde", config_class=HydeRetrieverConfig)
class HydeRetriever(RetrieverBase):
    """HydeRetriever is a retriever that rewrites the query before searching.

    The original paper is available at https://aclanthology.org/2023.acl-long.99/.
    The wrapped retriever performs query preprocessing and retrieval caching;
    this wrapper only rewrites input queries and forwards the resulting
    hypothetical passages to the wrapped retriever.
    """

    def __init__(
        self,
        cfg: HydeRetrieverConfig,
        retriever: RetrieverBase,
        generator: GeneratorProtocol,
    ) -> None:
        self.cfg = cfg
        self.retriever = retriever
        self.rewriter = HydeRewriter(
            generator=generator, task=cfg.task, language=cfg.language
        )
        return

    def search(
        self,
        query: Iterable[Any] | Any,
        disable_cache: bool = False,
        no_preprocess: bool = False,
        top_k: int = DEFAULT_TOP_K,
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        """Rewrite queries and delegate retrieval to the wrapped retriever.

        This method intentionally bypasses the base retriever's preprocessing
        and cache logic. The wrapped retriever receives ``disable_cache`` and
        ``no_preprocess`` unchanged and remains the only owner of those
        policies.

        :param query: Query or queries to rewrite.
        :param disable_cache: Whether to disable cache in the wrapped retriever.
        :param no_preprocess: Whether the wrapped retriever should skip query
            preprocessing.
        :param top_k: Number of retrieved contexts requested from the wrapped
            retriever.
        :param search_kwargs: Additional keyword arguments forwarded to the
            wrapped retriever.
        :return: Retrieved contexts returned by the wrapped retriever.
        """
        if isinstance(query, str):
            query = [query]
        elif isinstance(query, Iterable):
            query = list(query)
        else:
            query = [query]

        query = [str(q) for q in query]
        if not query:
            return []

        batch_size = max(1, self.cfg.batch_size)
        delegated_kwargs = dict(search_kwargs)
        delegated_kwargs["top_k"] = top_k
        delegated_kwargs["disable_cache"] = disable_cache
        delegated_kwargs["no_preprocess"] = no_preprocess

        results: list[list[RetrievedContext]] = []
        for start in range(0, len(query), batch_size):
            batch = query[start : start + batch_size]
            results.extend(self._search(batch, **dict(delegated_kwargs)))
        return results

    def _search(
        self,
        query: list[str],
        **search_kwargs,
    ) -> list[list[RetrievedContext]]:
        new_query = self.rewriter.rewrite(query)
        disable_cache = search_kwargs.pop("disable_cache", False)
        no_preprocess = search_kwargs.pop("no_preprocess", False)
        top_k = search_kwargs.pop("top_k", DEFAULT_TOP_K)
        return self.retriever.search(
            new_query,
            disable_cache=disable_cache,
            no_preprocess=no_preprocess,
            top_k=top_k,
            **search_kwargs,
        )

    @property
    def fields(self) -> list[str]:
        """Return fields produced by the wrapped retriever."""
        return self.retriever.fields
