from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

ItemTypeI = TypeVar("ItemTypeI")
ItemTypeM = TypeVar("ItemTypeM")
ItemTypeChain = TypeVar("ItemTypeChain")
ItemTypeConcat = TypeVar("ItemTypeConcat")


class IterableDataset(Iterable[ItemTypeI], Generic[ItemTypeI]):
    """IterableDataset is a dataset that can be iterated over.

    The subclasses of IterableDataset should implement the following methods:
    - __iter__(self) -> Iterator[ItemTypeI]: return an iterator over the items in the dataset.
    """

    def __add__(
        self, other: "IterableDataset[ItemTypeI]"
    ) -> "IterableDataset[ItemTypeI]":
        return ChainDataset(self, other)


class MappingDataset(Generic[ItemTypeM]):
    """MappingDataset is a dataset that can be indexed by integers.

    The subclasses of MappingDataset should implement the following methods:
    - __getitem__(self, index: int) -> ItemTypeM: return the item at the given index.
    - __len__(self) -> int: return the number of items in the dataset.

    The following methods are implemented automatically:
    - __iter__(self) -> Iterator[ItemTypeM]: return an iterator over the items in the dataset.
    - __contains__(self, key: int) -> bool: return whether the dataset contains the given index.
    - get(self, index: int, default=None) -> ItemTypeM: return the item at the given index or the default value.
    """

    def __add__(
        self, other: "MappingDataset[ItemTypeM]"
    ) -> "MappingDataset[ItemTypeM]":
        return ConcatDataset(self, other)

    def __contains__(self, key: int) -> bool:
        return 0 <= key < len(self)

    def __iter__(self) -> Iterator[ItemTypeM]:
        for i in range(len(self)):
            yield self[i]

    def get(self, index: int, default: Any = None) -> ItemTypeM:
        if 0 <= index < len(self):
            return self[index]
        return default


class ChainDataset(IterableDataset[ItemTypeChain]):
    def __init__(self, *datasets: IterableDataset):
        self.datasets = datasets
        return

    def __iter__(self) -> Iterator[ItemTypeChain]:
        for dataset in self.datasets:
            yield from dataset
        return


class ConcatDataset(MappingDataset[ItemTypeConcat]):
    def __init__(self, *datasets: MappingDataset):
        self.datasets = datasets
        return

    def __getitem__(self, index: int) -> ItemTypeConcat:
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError(f"Index {index} out of range.")

    def __iter__(self) -> Iterator[ItemTypeConcat]:
        for dataset in self.datasets:
            yield from dataset
        return

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)
