import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from tenacity import RetryCallState, retry, stop_after_attempt, wait_fixed

from kylin.utils import Choices

from .retriever_base import LocalRetriever, LocalRetrieverConfig, RetrievedContext

logger = logging.getLogger("TypesenseRetriever")


def _save_error_state(retry_state: RetryCallState) -> Exception:
    args = {
        "args": retry_state.args,
        "kwargs": retry_state.kwargs,
    }
    with open("typesense_retriever_error_state.json", "w") as f:
        json.dump(args, f)
    raise retry_state.outcome.exception()


@dataclass
class TypesenseRetrieverConfig(LocalRetrieverConfig): ...


class TypesenseRetriever(LocalRetriever):
    name = "Typesense"

    def __init__(self, cfg: TypesenseRetrieverConfig) -> None:
        super().__init__(cfg)
        return

    def add_passages(self, passages: Iterable[dict[str, str]] | list[str]):
        return

    def search_batch(
        self,
        query: list[str],
        top_k: int = 10,
        **search_kwargs,
    ) -> list[RetrievedContext]:
        return

    def clean(self) -> None:
        return

    def close(self) -> None:
        return

    def __len__(self) -> int:
        return

    @property
    def indices(self) -> list[str]:
        return

    @property
    def fingerprint(self) -> str:
        return

    def _form_results(
        self, query: list[str], responses: list[dict] | None
    ) -> list[list[RetrievedContext]]:
        return
