from abc import ABC, abstractmethod


class Retriever(ABC):
    @abstractmethod
    def add_passages(
        self,
        passages: list[dict[str, str]] | list[str],
        source: str = None,
    ):
        """
        Add passages to the retriever database
        """
        return

    @abstractmethod
    def search(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[dict[str, str | list]]:
        return

    @abstractmethod
    def close(self):
        return

    @property
    @abstractmethod
    def __len__(self):
        return
