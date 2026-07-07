from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import pytest

from flexrag.retrievers.backends.elastic import (
    INTERNAL_CONTEXT_ID,
    INTERNAL_META_CONTEXT_ID,
    INTERNAL_TEXT,
    INTERNAL_VECTOR,
    INTERNAL_VIEW,
)


class FakeIndicesClient:
    def __init__(self, parent: "FakeElasticClient") -> None:
        self.parent = parent

    async def exists(self, index: str) -> bool:
        return index in self.parent.docs

    async def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.parent.docs.setdefault(index, {})
        self.parent.created_mappings[index] = body
        return {"acknowledged": True}

    async def delete(self, index: str) -> dict[str, Any]:
        self.parent.docs.pop(index, None)
        self.parent.created_mappings.pop(index, None)
        return {"acknowledged": True}


class FakeElasticClient:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, dict[str, Any]]] = {}
        self.created_mappings: dict[str, dict[str, Any]] = {}
        self.msearch_bodies: list[list[dict[str, Any]]] = []
        self.indices = FakeIndicesClient(self)

    async def bulk(
        self,
        operations: list[dict[str, Any]],
        index: str | None = None,
    ) -> dict[str, Any]:
        items = []
        for meta, doc in zip(operations[0::2], operations[1::2]):
            target = meta["index"].get("_index", index)
            doc_id = meta["index"]["_id"]
            self.docs.setdefault(target, {})[doc_id] = dict(doc)
            items.append({"index": {"_id": doc_id, "status": 201}})
        return {"errors": False, "items": items}

    async def msearch(self, body: list[dict[str, Any]], **_: Any) -> dict[str, Any]:
        self.msearch_bodies.append(body)
        return {
            "responses": [
                {"status": 200, "hits": {"hits": self._search(header["index"], query)}}
                for header, query in zip(body[0::2], body[1::2])
            ]
        }

    async def search(self, index: str, body: dict[str, Any], **_: Any) -> dict[str, Any]:
        response = {"hits": {"hits": self._search(index, body)}}
        aggs = self._aggs(index, body)
        if aggs:
            response["aggregations"] = aggs
        return response

    async def close(self) -> None:
        return

    def _search(self, index: str, body: dict[str, Any]) -> list[dict[str, Any]]:
        query_text = _query_text(body)
        query_vector = body.get("knn", {}).get("query_vector")
        view = _term(body, INTERNAL_VIEW)
        context_id = _term(body, INTERNAL_CONTEXT_ID)
        scored = []
        for doc_id, doc in self.docs.get(index, {}).items():
            if view is not None and doc.get(INTERNAL_VIEW) != view:
                continue
            if context_id is not None and doc.get(INTERNAL_CONTEXT_ID) != context_id:
                continue
            if context_id is not None:
                score = 1.0
            elif query_vector is not None:
                if INTERNAL_VECTOR not in doc:
                    continue
                score = _dot(query_vector, doc[INTERNAL_VECTOR])
            else:
                score = _lexical_score(query_text, str(doc.get(INTERNAL_TEXT, "")))
                if score <= 0:
                    continue
            scored.append((score, doc_id, doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        size = int(body.get("size", 10))
        return [
            {"_id": doc_id, "_source": doc, "_score": score}
            for score, doc_id, doc in scored[:size]
        ]

    def _aggs(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        composite = (body.get("aggs") or {}).get("contexts", {}).get("composite")
        if not isinstance(composite, dict):
            return {}
        view = _term(body, INTERNAL_VIEW)
        counts: dict[str, int] = defaultdict(int)
        for doc in self.docs.get(index, {}).values():
            context_id = doc.get(INTERNAL_CONTEXT_ID)
            if context_id is None or context_id == INTERNAL_META_CONTEXT_ID:
                continue
            if view is not None and doc.get(INTERNAL_VIEW) != view:
                continue
            counts[str(context_id)] += 1
        buckets = [
            {"key": {"context_id": key}, "doc_count": count}
            for key, count in sorted(counts.items())
        ]
        return {"contexts": {"buckets": buckets[: int(composite.get("size", 10))]}}


@pytest.fixture
def fake_elastic_client() -> FakeElasticClient:
    return FakeElasticClient()


def _query_text(body: dict[str, Any]) -> str:
    for item in body.get("query", {}).get("bool", {}).get("must", []):
        if "match" in item:
            return str(next(iter(item["match"].values())))
    return ""


def _term(body: dict[str, Any], field: str) -> str | None:
    query = body.get("query", {})
    if "term" in query and field in query["term"]:
        return str(query["term"][field])
    filters = list(query.get("bool", {}).get("filter", []))
    knn_filter = body.get("knn", {}).get("filter", [])
    filters.extend(knn_filter if isinstance(knn_filter, list) else [knn_filter])
    for item in filters:
        term = item.get("term") if isinstance(item, dict) else None
        if isinstance(term, dict) and field in term:
            return str(term[field])
    return None


def _lexical_score(query: str, text: str) -> float:
    query_terms = query.lower().replace("-", " ").split()
    text_terms = text.lower().replace("-", " ").split()
    return float(sum((Counter(query_terms) & Counter(text_terms)).values()))


def _dot(query: list[float], vector: list[float]) -> float:
    return sum(float(q) * float(v) for q, v in zip(query, vector))
