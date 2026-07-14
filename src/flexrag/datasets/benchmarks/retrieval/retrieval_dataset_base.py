from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from flexrag.common import Context, RetrievedContext

from ...core import IRSample, MappingDataset, RankingSample
from ...corpora.corpus_dataset import IterableCorpus, MappingCorpus


class RetrievalDatasetBase(MappingDataset[IRSample | RankingSample]):
    """Sample-centric base class for retrieval benchmarks."""

    _corpus: Optional[IterableCorpus | MappingCorpus] = None

    @property
    def corpus(self) -> Optional[IterableCorpus | MappingCorpus]:
        return self._corpus

    def _get_context(
        self,
        ctx_id: str,
    ) -> Context:
        if (
            self.corpus is not None
            and hasattr(self.corpus, "contexts")
            and ctx_id in self.corpus.contexts
        ):
            context = self.corpus.contexts[ctx_id]
            return Context(
                context_id=context.context_id,
                data=dict(context.data),
                source=context.source,
                metadata=dict(context.metadata),
            )
        return Context(context_id=ctx_id)

    def _build_candidate(
        self,
        question: str,
        candidate: dict[str, Any],
    ) -> RetrievedContext:
        context = self._get_context(candidate["ctx_id"])
        return RetrievedContext(
            **asdict(context),
            query=question,
            score=candidate.get("score"),
            retriever=candidate.get("retriever"),
        )

    def build_sample(
        self,
        *,
        question: str,
        question_id: str,
        qrels: dict[str, float],
        candidates: list[dict[str, Any]] | None = None,
        metadata: dict | None = None,
    ) -> IRSample | RankingSample:
        relevant_ctxs: list[Context] = []
        for ctx_id, relevance in qrels.items():
            if relevance > 0:
                relevant_ctxs.append(self._get_context(ctx_id))

        if candidates:
            candidate_ctxs: list[RetrievedContext] = []
            for candidate in candidates:
                candidate_ctxs.append(self._build_candidate(question, candidate))
            return RankingSample(
                question=question,
                question_id=question_id,
                contexts=relevant_ctxs,
                qrels=dict(qrels),
                candidates=candidate_ctxs,
                metadata=metadata,
            )

        return IRSample(
            question=question,
            question_id=question_id,
            contexts=relevant_ctxs,
            qrels=dict(qrels),
            metadata=metadata,
        )
