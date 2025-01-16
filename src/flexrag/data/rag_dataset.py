from dataclasses import dataclass, field
from typing import Iterator, Optional

from .line_delimited_dataset import LineDelimitedDataset


@dataclass
class RAGTestData:
    question: str
    golden_contexts: Optional[list[str]] = None
    golden_answers: Optional[list[str]] = None
    meta_data: dict = field(default_factory=dict)


class RAGTestIterableDataset(LineDelimitedDataset):
    def __iter__(self) -> Iterator[RAGTestData]:
        for data in super().__iter__():
            formatted_data = RAGTestData(
                question=data.pop("question"),
                golden_contexts=data.pop("golden_contexts", None),
                golden_answers=data.pop("golden_answers", None),
            )
            formatted_data.meta_data = data.pop("meta_data", {})
            formatted_data.meta_data.update(data)
            yield formatted_data
