from dataclasses import dataclass
from typing import Iterator, Optional

from .dataset import IterableDataset


@dataclass
class RAGTestData:
    question: str
    golden_contexts: Optional[list[str]] = None
    golden_answers: Optional[list[str]] = None
    meta_data: Optional[dict] = None


class RAGTestIterableDataset(IterableDataset):
    def __iter__(self) -> Iterator[RAGTestData]:
        for data in super().__iter__():
            yield RAGTestData(
                question=data["question"],
                golden_contexts=data.get("golden_contexts"),
                golden_answers=data.get("golden_answers"),
                meta_data=data.get("meta_data"),
            )


@dataclass
class RetrievalTestData:
    question: str
    contexts: list[str]
    golden_contexts: list[str]
    meta_data: Optional[dict] = None


class RetrievalTestIterableDataset(IterableDataset):
    def __iter__(self) -> Iterator[RetrievalTestData]:
        for data in super().__iter__():
            yield RetrievalTestData(
                question=data["question"],
                golden_contexts=data["contexts"],
                golden_answers=data["golden_contexts"],
                meta_data=data.get("meta_data"),
            )
