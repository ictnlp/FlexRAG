from abc import abstractmethod
from collections.abc import Iterator, Mapping
from functools import cached_property

from flexrag.common.dataclasses import Context

from ...core import IRSample, MappingDataset


class RetrievalDatasetBase(MappingDataset[IRSample]):
    """Base class for Information Retrieval (IR) datasets.

    This class provides a unified interface for accessing IR datasets, which typically consist of:
    1. A corpus of documents (contexts).
    2. A set of queries.
    3. Relevance judgments (qrels) linking queries to relevant documents.

    It inherits from `MappingDataset[IREvalData]`, allowing iteration and random access
    to `IREvalData` objects. Each `IREvalData` contains a query and its associated
    ground-truth contexts (gold contexts).

    Subclasses must implement the following abstract properties to define the data source:

        >>> @property
        >>> def _contexts(self) -> Mapping[str, Context]:
        >>>     # Return a mapping from context_id to Context object
        >>>     ...

        >>> @property
        >>> def _queries(self) -> Mapping[str, str]:
        >>>     # Return a mapping from query_id to query text
        >>>     ...

        >>> @property
        >>> def _qrels(self) -> Mapping[str, Mapping[str, float]]:
        >>>     # Return a mapping from query_id to a set of relevant context_ids and their relevance scores
        >>>     ...

    The class automatically implements the following functionality:

        >>> # Retrieve a query string by its ID
        >>> def get_query(self, query_id: str) -> str: ...

        >>> # Retrieve a Context object by its ID
        >>> def get_context(self, context_id: str) -> Context: ...

        >>> # Get the total number of queries
        >>> @property
        >>> def queries_count(self) -> int: ...

        >>> # Get the total number of documents in the corpus
        >>> @property
        >>> def contexts_count(self) -> int: ...

        >>> # Iterator over all query IDs
        >>> @property
        >>> def query_ids(self) -> Iterator[str]: ...

        >>> # Iterator over all context IDs
        >>> @property
        >>> def context_ids(self) -> Iterator[str]: ...
    """

    @property
    @abstractmethod
    def _contexts(self) -> Mapping[str, Context]:
        """The contexts of the dataset."""
        return

    @property
    @abstractmethod
    def _queries(self) -> Mapping[str, str]:
        """The queries of the dataset."""
        return

    @property
    @abstractmethod
    def _qrels(self) -> Mapping[str, Mapping[str, float]]:
        """The qrels of the dataset."""
        return

    @cached_property
    def _qids(self) -> list[str]:
        """The index of the queries in the qrels."""
        return sorted(self._qrels.keys())

    def get_query(self, query_id: str) -> str:
        """Get the query by query id.

        :param query_id: The id of the query.
        :type query_id: str
        :return: The query string.
        :rtype: str
        """
        return self._queries[query_id]

    @property
    def query_ids(self) -> Iterator[str]:
        """Get all query ids in the dataset.

        :return: An iterator of query ids.
        :rtype: Iterator[str]
        """
        yield from self._queries

    @property
    def queries_count(self) -> int:
        """The number of queries in the dataset."""
        return len(self._queries)

    def get_context(self, context_id: str) -> Context:
        """Get the context by context id.

        :param context_id: The id of the context.
        :type context_id: str
        :return: The context.
        :rtype: Context
        """
        return self._contexts[context_id]

    @property
    def context_ids(self) -> Iterator[str]:
        """Get all context ids in the dataset.

        :return: An iterator of context ids.
        :rtype: Iterator[str]
        """
        yield from self._contexts

    @property
    def contexts_count(self) -> int:
        """The number of context documents in the dataset."""
        return len(self._contexts)

    def __len__(self) -> int:
        """The number of queries in the qrels."""
        return len(self._qids)

    def get_item(self, index: int) -> IRSample:
        qid = self._qids[index]
        query = self._queries[qid]
        relevant_ctxs = []
        hard_negatives = []
        for ctx_id, relevance in self._qrels[qid].items():
            if relevance > 0:
                relevant_ctxs.append(ctx_id)
            else:
                hard_negatives.append(ctx_id)
        rels = [
            self._contexts.get(ctx_id, Context(context_id=ctx_id))
            for ctx_id in relevant_ctxs
        ]
        negs = [
            self._contexts.get(ctx_id, Context(context_id=ctx_id))
            for ctx_id in hard_negatives
        ]
        return IRSample(
            question=query,
            question_id=qid,
            contexts=rels,
            hard_negatives=negs,
        )
