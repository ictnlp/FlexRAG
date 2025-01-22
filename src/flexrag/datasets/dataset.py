from collections.abc import Iterable, Iterator
from typing import Any, Generic, TypeVar

ItemTypeI = TypeVar("ItemTypeI")
ItemTypeM = TypeVar("ItemTypeM")
ItemTypeChain = TypeVar("ItemTypeChain")
ItemTypeConcat = TypeVar("ItemTypeConcat")


class IterableDataset(Iterable[ItemTypeI], Generic[ItemTypeI]):
    r"""IterableDataset is a BaseClass for datasets that can be iterated over.

    The subclasses of IterableDataset should implement the following methods:

        >>> # return an iterator over the items in the dataset.
        >>> def __iter__(self) -> Iterator[ItemTypeI]: ...

    The following methods are implemented automatically:

        >>> # concatenate multiple IterableDatasets.
        >>> def __add__(self, other: IterableDataset[ItemTypeI]) -> IterableDataset[ItemTypeI]: ...

    For example:

        >>> class MyDataset(IterableDataset[int]):
        ...     def __init__(self, n: int):
        ...         self.n = n
        ...         return
        ...
        ...     def __iter__(self) -> Iterator[int]:
        ...         for i in range(self.n):
        ...             yield i
        ...
        >>> dataset = MyDataset(3)
        >>> # Iterate over the dataset.
        >>> for item in dataset:
        ...     print(item)
    """

    def __add__(
        self, other: "IterableDataset[ItemTypeI]"
    ) -> "IterableDataset[ItemTypeI]":
        return ChainDataset(self, other)


class MappingDataset(Generic[ItemTypeM]):
    r"""MappingDataset is a BaseClass for datasets that can be indexed by integers.

    The subclasses of MappingDataset should implement the following methods:

        >>> # retrun the item at the given index.
        >>> def __getitem__(self, index: int) -> ItemTypeM: ...
        >>> # return the number of items in the dataset.
        >>> def __len__(self) -> int: ...

    The following methods are implemented automatically:

        >>> # concatenate multiple MappingDatasets.
        >>> def __add__(self, other: MappingDataset[ItemTypeM]) -> MappingDataset[ItemTypeM]: ...
        >>> # return whether the dataset contains the given index.
        >>> def __contains__(self, key: int) -> bool: ...
        >>> # return an iterator over the items in the dataset.
        >>> def __iter__(self) -> Iterator[ItemTypeM]: ...

    For example:

        >>> class MyDataset(MappingDataset[int]):
        ...     def __init__(self, n: int):
        ...         self.n = n
        ...         return
        ...
        ...     def __getitem__(self, index: int) -> int:
        ...         if 0 <= index < self.n:
        ...             return index
        ...         raise IndexError(f"Index {index} out of range.")
        ...
        ...     def __len__(self) -> int:
        ...         return self.n
        ...
        >>> dataset = MyDataset(3)
        >>> for i in range(len(dataset)):
        ...     print(dataset[i])
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
    """ChainDataset concatenates multiple IterableDatasets."""

    def __init__(self, *datasets: IterableDataset):
        self.datasets = datasets
        return

    def __iter__(self) -> Iterator[ItemTypeChain]:
        for dataset in self.datasets:
            yield from dataset
        return


class ConcatDataset(MappingDataset[ItemTypeConcat]):
    """ConcatDataset concatenates multiple MappingDatasets."""

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
