from .retriever_base import (
    LocalRetriever,
    LocalRetrieverConfig,
    Retriever,
    RetrieverConfig,
)
from .dense_retriever import DenseRetriever, DenseRetrieverConfig
from .bm25_retriever import BM25Retriever, BM25RetrieverConfig
from .web_retriever import (
    WebRetriever,
    WebRetrieverConfig,
    BingRetriever,
    BingRetrieverConfig,
    DuckDuckGoRetriever,
    DuckDuckGoRetrieverConfig,
)

__all__ = [
    "LocalRetriever",
    "LocalRetrieverConfig",
    "Retriever",
    "RetrieverConfig",
    "DenseRetriever",
    "DenseRetrieverConfig",
    "BM25Retriever",
    "BM25RetrieverConfig",
    "WebRetriever",
    "WebRetrieverConfig",
    "BingRetriever",
    "BingRetrieverConfig",
    "DuckDuckGoRetriever",
    "DuckDuckGoRetrieverConfig",
    "add_args_for_retriever",
    "load_retrievers",
]
