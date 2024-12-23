from typing import Iterable, Iterator


class Dataset(Iterable):
    def __add__(self, other: "Dataset") -> "Dataset":
        return ConcateDataset(self, other)


class ConcateDataset(Dataset):
    def __init__(self, *datasets: Dataset):
        self.datasets = datasets
        return

    def __iter__(self) -> Iterator[dict]:
        for dataset in self.datasets:
            yield from dataset
        return
