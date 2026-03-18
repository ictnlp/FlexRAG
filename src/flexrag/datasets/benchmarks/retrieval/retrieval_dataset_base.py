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
        *,
        query: str | None = None,
        score: float | None = None,
        retriever: str | None = None,
    ) -> Context:
        if (
            self.corpus is not None
            and hasattr(self.corpus, "contexts")
            and ctx_id in self.corpus.contexts
        ):
            context = self.corpus.contexts[ctx_id]
            if query is None and score is None and retriever is None:
                return context
            return RetrievedContext(
                **asdict(context),
                query=query,
                score=score,
                retriever=retriever,
            )
        if query is None and score is None and retriever is None:
            return Context(context_id=ctx_id)
        return RetrievedContext(
            context_id=ctx_id,
            query=query,
            score=score,
            retriever=retriever,
        )

    def build_sample(
        self,
        *,
        question: str,
        question_id: str,
        qrels: dict[str, float],
        candidates: list[dict[str, Any]] | None = None,
        meta_data: dict | None = None,
    ) -> IRSample | RankingSample:
        relevant_ctxs: list[RetrievedContext] = []
        for ctx_id, relevance in qrels.items():
            if relevance > 0:
                ctx = self._get_context(ctx_id, query=question, score=relevance)
                assert isinstance(ctx, RetrievedContext)
                relevant_ctxs.append(ctx)

        if candidates:
            candidate_ctxs: list[Context] = []
            for candidate in candidates:
                ctx = self._get_context(
                    candidate["ctx_id"],
                    query=question,
                    score=candidate.get("score"),
                    retriever=candidate.get("retriever"),
                )
                candidate_ctxs.append(ctx)
            return RankingSample(
                question=question,
                question_id=question_id,
                contexts=relevant_ctxs,
                candidates=candidate_ctxs,
                meta_data=meta_data,
            )

        return IRSample(
            question=question,
            question_id=question_id,
            contexts=relevant_ctxs,
            meta_data=meta_data,
        )
