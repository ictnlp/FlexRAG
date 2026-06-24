from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Generic, TypeVar, overload

from flexrag.common import Register

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


class MappingDataset(Mapping[int, ItemTypeM], Generic[ItemTypeM]):
    r"""MappingDataset is a BaseClass for datasets that can be indexed by integers.

    The subclasses of MappingDataset should implement the following methods:

        >>> # retrun the item at the given index.
        >>> def get_item(self, index: int) -> ItemTypeM: ...

        >>> # return the number of items in the dataset.
        >>> def __len__(self) -> int: ...

    The following methods are implemented automatically:

        >>> # concatenate multiple MappingDatasets.
        >>> def __add__(self, other: MappingDataset[ItemTypeM]) -> MappingDataset[ItemTypeM]: ...

        >>> # return whether the dataset contains the given index.
        >>> def __contains__(self, key: int) -> bool: ...

        >>> # return an iterator over the items in the dataset.
        >>> def __iter__(self) -> Iterator[ItemTypeM]: ...

        >>> # get the item at the given index, or return a SubDataset if a slice is given.
        >>> def __getitem__(self, index: int | slice) -> ItemTypeM | MappingDataset[ItemTypeM]: ...

    For example:

        >>> class MyDataset(MappingDataset[int]):
        ...     def __init__(self, n: int):
        ...         self.n = n
        ...         return
        ...
        ...     def get_item(self, index: int) -> int:
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

    @overload
    def __getitem__(self, index: int) -> ItemTypeM:
        """Get the item from the dataset by index."""

    @overload
    def __getitem__(self, index: slice) -> MappingDataset[ItemTypeM]:
        """Get a subset of the MappingDataset by slice."""

    def __getitem__(self, index: int | slice) -> ItemTypeM | MappingDataset[ItemTypeM]:
        if isinstance(index, slice):
            return SubDataset(self, index)
        return self.get_item(index)

    @abstractmethod
    def get_item(self, index: int) -> ItemTypeM:
        """Get the item at the given index.

        :param index: The index of the item.
        :type index: int
        :return: The item at the given index.
        :rtype: ItemTypeM
        """
        return


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

    def get_item(self, index: int) -> ItemTypeConcat:
        original_index = index
        if index < 0:
            index += len(self)
        if index < 0:
            raise IndexError(f"Index {original_index} out of range.")

        for dataset in self.datasets:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError(f"Index {original_index} out of range.")

    def __iter__(self) -> Iterator[ItemTypeConcat]:
        for dataset in self.datasets:
            yield from dataset
        return

    def __len__(self) -> int:
        return sum(len(dataset) for dataset in self.datasets)


class SubDataset(MappingDataset[ItemTypeM]):
    """SubDataset is a dataset that contains a subset of another dataset.
    The SubDataset is a view of the original dataset, and does not copy the data.

    :param dataset: The original dataset.
    :type dataset: MappingDataset[ItemTypeM]
    :param indices: The indices of the subset.
    :type indices: list[int] | slice
    """

    def __init__(
        self, dataset: MappingDataset[ItemTypeM], indices: list[int] | slice
    ) -> None:
        self.dataset = dataset
        if isinstance(indices, slice):
            self.indices = list(range(*indices.indices(len(dataset))))
        else:
            self.indices = indices
        return

    def __getitem__(self, index: int | slice) -> ItemTypeM | MappingDataset[ItemTypeM]:
        # re-implement to prevent recursion
        if isinstance(index, slice):
            subset_slice = self.indices[index]
            clone = SubDataset(self.dataset, subset_slice)
            return clone
        return self.dataset[self.indices[index]]

    def get_item(self, index: int) -> ItemTypeM:
        return self.dataset.get_item(self.indices[index])

    def __len__(self) -> int:
        return len(self.indices)

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying dataset."""
        return getattr(self.dataset, name)


DATASETS = Register[MappingDataset | IterableDataset]("datasets")
