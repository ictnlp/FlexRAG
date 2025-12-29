from abc import abstractmethod
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from functools import cached_property
from typing import Any

from flexrag.common import Context, RetrievedContext

from ...core import IRSample, MappingDataset, RankingSample


class RetrievalDatasetBase(MappingDataset[IRSample | RankingSample]):
    """Base class for Information Retrieval (IR) datasets.

    This class provides a unified interface for accessing IR datasets, which typically consist of:
    1. A corpus of documents (contexts).
    2. A set of queries.
    3. Relevance judgments (qrels) linking queries to relevant documents.
    4. A list of candidates for each query (optional, for ranking tasks).

    It inherits from `MappingDataset[IRSample | RankingSample]`, allowing iteration and random access
    to `IRSample` or `RankingSample` objects. Each object contains a query and its associated
    ground-truth contexts (gold contexts).

    Subclasses must implement the following abstract properties to define the data source:

        >>> @property
        >>> def contexts(self) -> Mapping[str, Context]:
        >>>     # Return a mapping from context_id to Context object
        >>>     ...

        >>> @property
        >>> def queries(self) -> Mapping[str, str]:
        >>>     # Return a mapping from query_id to query text
        >>>     ...

        >>> @property
        >>> def qrels(self) -> Mapping[str, Mapping[str, float]]:
        >>>     # Return a mapping from query_id to a set of relevant context_ids and their relevance scores
        >>>     ...

    The class automatically implements the following functionality:

        >>> # Iterator over all query IDs
        >>> @cached_property
        >>> def query_ids(self) -> list[str]: ...

        >>> # Iterator over all context IDs
        >>> @property
        >>> def context_ids(self) -> Iterator[str]: ...
    """

    @property
    @abstractmethod
    def contexts(self) -> Mapping[str, Context]:
        """The contexts of the dataset."""
        return

    @property
    @abstractmethod
    def queries(self) -> Mapping[str, str]:
        """The queries of the dataset."""
        return

    @property
    @abstractmethod
    def qrels(self) -> Mapping[str, Mapping[str, float]]:
        """The qrels of the dataset."""
        return

    @property
    def candidates(self) -> Mapping[str, Mapping[str, Any]]:
        """The candidates for each query in the dataset."""
        return {}

    @cached_property
    def query_ids(self) -> list[str]:
        """The index of the queries in the qrels."""
        return sorted(self.qrels.keys())

    @property
    def context_ids(self) -> Iterator[str]:
        """Get all context ids in the dataset.

        :return: An iterator of context ids.
        :rtype: Iterator[str]
        """
        yield from self.contexts.keys()

    def __len__(self) -> int:
        """The number of queries in the qrels."""
        return len(self.query_ids)

    def get_item(self, index: int) -> IRSample | RankingSample:
        qid = self.query_ids[index]
        query = self.queries[qid]

        # load relevant contexts
        relevant_ctxs: list[RetrievedContext] = []
        for ctx_id, relevance in self.qrels[qid].items():
            if relevance > 0:
                if ctx_id in self.contexts:
                    ctx = RetrievedContext(
                        **asdict(self.contexts[ctx_id]),
                        score=relevance,
                        query=query,
                    )
                else:
                    ctx = RetrievedContext(
                        context_id=ctx_id,
                        query=query,
                        score=relevance,
                    )
                relevant_ctxs.append(ctx)

        # load candidate contexts if available
        candidates: list[Context] = []
        for candidate in self.candidates.get(qid, []):
            ctx_id = candidate["ctx_id"]
            if ctx_id in self.contexts:
                ctx = RetrievedContext(
                    **asdict(self.contexts[ctx_id]),
                    query=query,
                    score=candidate.get("score"),
                    retriever=candidate.get("retriever"),
                )
            else:
                ctx = RetrievedContext(
                    context_id=ctx_id,
                    query=query,
                    score=candidate.get("score"),
                    retriever=candidate.get("retriever"),
                )
            candidates.append(ctx)

        if candidates:
            return RankingSample(
                question=query,
                question_id=qid,
                contexts=relevant_ctxs,
                candidates=candidates,
            )

        return IRSample(
            question=query,
            question_id=qid,
            contexts=relevant_ctxs,
        )
