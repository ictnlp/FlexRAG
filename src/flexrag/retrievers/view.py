from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from flexrag.common.dataclasses import Context

ViewMergeMethod = Literal["max", "mean", "sum", "concat"]
_MERGE_METHODS = {"max", "mean", "sum", "concat"}


@dataclass(frozen=True)
class ProjectedRow:
    """One row produced by applying a retrieval view to a context.

    :param context_id: Source context identifier.
    :param field: Source field name, or ``"__concat__"`` for concatenated text.
    :param content: Projected row content passed to the backend.
    """

    context_id: str
    field: str
    content: Any


@dataclass(frozen=True)
class RetrievalView:
    """Projection strategy from ``Context.data`` into backend rows.

    :param name: Stable view name persisted by backend artifacts.
    :param fields: Context data fields to project.
    :param merge_method: Row score aggregation method for multi-row contexts.
    """

    name: str
    fields: list[str]
    merge_method: ViewMergeMethod = "max"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("RetrievalView.name must be non-empty.")
        if not self.fields:
            raise ValueError("RetrievalView.fields must be non-empty.")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("RetrievalView.fields must be unique.")
        if self.merge_method not in _MERGE_METHODS:
            raise ValueError(f"Invalid merge_method: {self.merge_method}")
        return

    def project(self, context: Context) -> list[ProjectedRow]:
        """Project one context into zero or more backend rows.

        Missing fields are skipped. ``concat`` emits one text row and requires
        every selected field value to be a string.

        :param context: Context to project.
        :returns: Projected rows for fields available in the context.
        :raises TypeError: If ``concat`` sees non-text content.
        :raises ValueError: If the context has no id.
        """
        if context.context_id is None:
            raise ValueError("context_id is required for indexing.")
        selected = [
            (field, context.data[field])
            for field in self.fields
            if field in context.data
        ]
        if not selected:
            return []

        if self.merge_method == "concat":
            parts = []
            for field, value in selected:
                if not isinstance(value, str):
                    raise TypeError(
                        "merge_method='concat' only supports text fields; "
                        f"field {field!r} has type {type(value).__name__}."
                    )
                parts.append(f"{field}: {value}")
            return [
                ProjectedRow(
                    context_id=context.context_id,
                    field="__concat__",
                    content=" ".join(parts),
                )
            ]

        rows = []
        for field, value in selected:
            rows.append(
                ProjectedRow(
                    context_id=context.context_id,
                    field=field,
                    content=value,
                )
            )
        return rows

    def aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate row-level scores into one context-level score.

        :param scores: Non-empty row score list.
        :returns: Aggregated context score according to ``merge_method``.
        :raises ValueError: If ``scores`` is empty.
        """
        if not scores:
            raise ValueError("Cannot aggregate an empty score list.")
        match self.merge_method:
            case "max" | "concat":
                return max(scores)
            case "mean":
                return sum(scores) / len(scores)
            case "sum":
                return sum(scores)
            case _:
                raise ValueError(f"Invalid merge_method: {self.merge_method}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the view for backend artifacts.

        :returns: Plain dictionary containing name, fields, and merge method.
        """
        return {
            "name": self.name,
            "fields": list(self.fields),
            "merge_method": self.merge_method,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievalView":
        """Restore a retrieval view from artifact metadata.

        :param data: Serialized view dictionary.
        :returns: Restored retrieval view.
        """
        return cls(
            name=data["name"],
            fields=data["fields"],
            merge_method=data["merge_method"],
        )
