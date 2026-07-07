from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from .backends.base import Hit

MergeMethod = Literal["rrf", "linear"]


def normalize_merge_method(
    method: str | None,
    default: MergeMethod,
) -> MergeMethod:
    """Normalize a user-supplied hit merge method.

    :param method: Explicit merge method, or ``None`` to use ``default``.
    :param default: Default merge method.
    :returns: Normalized merge method.
    :raises ValueError: If the method is unknown.
    """
    normalized = method or default
    if normalized not in {"rrf", "linear"}:
        raise ValueError(f"Unknown merge method: {normalized}")
    return cast(MergeMethod, normalized)


def normalize_backend_weights(
    backend_names: Sequence[str],
    backend_weights: Mapping[str, float] | None,
) -> list[float]:
    """Normalize backend weights for a selected backend list.

    :param backend_names: Ordered backend names participating in merge.
    :param backend_weights: Optional unnormalized weight mapping.
    :returns: Weights aligned with ``backend_names`` and summing to one.
    :raises ValueError: If the provided weights have non-positive sum.
    """
    if not backend_names:
        return []
    if backend_weights is None:
        return [1.0 / len(backend_names)] * len(backend_names)
    weights = [float(backend_weights.get(name, 1.0)) for name in backend_names]
    total = sum(weights)
    if total <= 0:
        raise ValueError("backend_weights must have a positive sum.")
    return [weight / total for weight in weights]


def merge_hits(
    per_backend_results: list[list[list[Hit]]],
    *,
    backend_names: Sequence[str],
    weights: Sequence[float],
    top_k: int,
    merge_method: MergeMethod,
    rrf_base: int,
) -> list[list[Hit]]:
    """Merge per-backend hit lists query by query.

    :param per_backend_results: Backend results shaped as backend/query/hits.
    :param backend_names: Backend names aligned with ``per_backend_results``.
    :param weights: Normalized backend weights.
    :param top_k: Maximum merged hits per query.
    :param merge_method: Merge algorithm to use.
    :param rrf_base: RRF denominator base.
    :returns: One merged hit list per query.
    :raises ValueError: If backend result shapes are inconsistent.
    """
    if not per_backend_results:
        return []
    if len(per_backend_results) != len(backend_names):
        raise ValueError("backend_names length must match per_backend_results.")
    if len(per_backend_results) != len(weights):
        raise ValueError("weights length must match per_backend_results.")
    if top_k <= 0:
        return [[] for _ in range(len(per_backend_results[0]))]
    _validate_result_shape(per_backend_results)
    match merge_method:
        case "rrf":
            return _merge_rrf(
                per_backend_results,
                weights=weights,
                top_k=top_k,
                rrf_base=rrf_base,
            )
        case "linear":
            return _merge_linear(
                per_backend_results,
                weights=weights,
                top_k=top_k,
            )
        case _:
            raise ValueError(f"Unknown merge method: {merge_method}")


def _validate_result_shape(per_backend_results: list[list[list[Hit]]]) -> None:
    query_count = len(per_backend_results[0])
    for backend_results in per_backend_results:
        if len(backend_results) != query_count:
            raise ValueError("Each backend result must contain one list per query.")
    return


def _merge_rrf(
    per_backend_results: list[list[list[Hit]]],
    *,
    weights: Sequence[float],
    top_k: int,
    rrf_base: int,
) -> list[list[Hit]]:
    merged: list[list[Hit]] = []
    for query_idx in range(len(per_backend_results[0])):
        scores: dict[str, float] = defaultdict(float)
        first_hits: dict[str, Hit] = {}
        for backend_results, weight in zip(per_backend_results, weights):
            for rank, hit in enumerate(backend_results[query_idx], start=1):
                scores[hit.context_id] += weight / (rrf_base + rank)
                first_hits.setdefault(hit.context_id, hit)
        merged.append(_build_merged_hits(scores, first_hits, top_k=top_k))
    return merged


def _merge_linear(
    per_backend_results: list[list[list[Hit]]],
    *,
    weights: Sequence[float],
    top_k: int,
) -> list[list[Hit]]:
    merged: list[list[Hit]] = []
    for query_idx in range(len(per_backend_results[0])):
        scores: dict[str, float] = defaultdict(float)
        first_hits: dict[str, Hit] = {}
        for backend_results, weight in zip(per_backend_results, weights):
            hits = list(backend_results[query_idx])
            if not hits:
                continue
            raw_scores = [float(hit.score) for hit in hits]
            infimum = min(raw_scores)
            denominator = max(raw_scores) - infimum
            if denominator == 0:
                denominator = 1.0
            for hit, raw_score in zip(hits, raw_scores):
                scores[hit.context_id] += ((raw_score - infimum) / denominator) * weight
                first_hits.setdefault(hit.context_id, hit)
        merged.append(_build_merged_hits(scores, first_hits, top_k=top_k))
    return merged


def _build_merged_hits(
    scores: dict[str, float],
    first_hits: dict[str, Hit],
    *,
    top_k: int,
) -> list[Hit]:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        Hit(
            context_id=context_id,
            score=float(score),
            backend=first_hits[context_id].backend,
            view=first_hits[context_id].view,
            context=first_hits[context_id].context,
        )
        for context_id, score in ordered[:top_k]
    ]
