from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from flexrag.common import Context
from flexrag.retrievers import (
    FlexRetriever,
    Hit,
    RetrievalView,
    SyncCollectionBackendBase,
)


class StaticBackend(SyncCollectionBackendBase):
    requires_context_store = False
    is_addable = True

    def __init__(self, name: str, ranked: list[tuple[str, float]]) -> None:
        super().__init__(RetrievalView(name, ["text"]))
        self.ranked = ranked
        self.calls: list[tuple[int, dict[str, Any] | None]] = []

    def rebuild(self, contexts: Iterable[Context]) -> None:
        return

    def search_hits(
        self,
        queries: list[Any],
        top_k: int,
        *,
        search_options: dict[str, Any] | None = None,
    ) -> list[list[Hit]]:
        self.calls.append((top_k, search_options))
        view = self._require_view()
        hits = [
            Hit(
                context_id=context_id,
                score=score,
                backend="",
                view=view.name,
                context=Context(context_id=context_id, data={"text": context_id}),
            )
            for context_id, score in self.ranked[:top_k]
        ]
        return [list(hits) for _ in queries]

    def clear(self) -> None:
        return

    def count(self) -> int:
        return len({context_id for context_id, _ in self.ranked})


def ids(results: list[list[Hit]]) -> list[str]:
    return [hit.context_id for hit in results[0]]


def test_rrf_uses_candidate_k() -> None:
    left = StaticBackend("left", [("doc-a", 10.0), ("doc-c", 1.0)])
    right = StaticBackend("right", [("doc-b", 10.0), ("doc-c", 1.0)])
    retriever = FlexRetriever.from_backends({"left": left, "right": right})
    assert ids(retriever.search_hits("q", top_k=1)) == ["doc-a"]
    assert ids(retriever.search_hits("q", top_k=1, candidate_k=2)) == ["doc-c"]
    assert left.calls[-1][0] == 2
    assert right.calls[-1][0] == 2
    assert retriever.search("q", top_k=1, candidate_k=2)[0][0].data == {
        "text": "doc-c"
    }


def test_backend_weights_and_linear_merge_change_order() -> None:
    weak = StaticBackend("weak", [("doc-a", 1.0), ("doc-b", 0.0)])
    strong = StaticBackend("strong", [("doc-b", 1.0), ("doc-a", 0.0)])
    retriever = FlexRetriever.from_backends({"weak": weak, "strong": strong})
    assert ids(
        retriever.search_hits(
            "q",
            top_k=1,
            candidate_k=2,
            backend_weights={"weak": 0.1, "strong": 0.9},
        )
    ) == ["doc-b"]

    left = StaticBackend("left", [("doc-a", 1.0), ("doc-b", 100.0)])
    right = StaticBackend("right", [("doc-c", 5.0)])
    retriever = FlexRetriever.from_backends({"left": left, "right": right})
    assert ids(
        retriever.search_hits("q", top_k=1, candidate_k=2, merge_method="linear")
    ) == ["doc-b"]


def test_single_backend_options_and_empty_results() -> None:
    backend = StaticBackend("only", [("doc-a", 1.0), ("doc-b", 0.0)])
    retriever = FlexRetriever.from_backends({"only": backend})
    assert ids(
        retriever.search_hits(
            "q",
            top_k=1,
            candidate_k=10,
            backend_search_options={"only": {"where": "x"}},
        )
    ) == ["doc-a"]
    assert backend.calls[-1] == (1, {"where": "x"})
    assert retriever.search_hits("q", top_k=0) == [[]]
    assert FlexRetriever.from_backends({}).search_hits(["a", "b"], top_k=3) == [
        [],
        [],
    ]
