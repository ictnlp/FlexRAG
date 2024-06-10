from .retriever_base import LocalRetriever, Retriever
from .dense_retriever import DenseRetriever
from .bm25_retriever import BM25Retriever
from .web_retriever import WebRetriever, BingRetriever, DuckDuckGoRetriever
from .loader import add_args_for_retriever, load_retrievers

__all__ = [
    "LocalRetriever",
    "Retriever",
    "DenseRetriever",
    "BM25Retriever",
    "WebRetriever",
    "BingRetriever",
    "DuckDuckGoRetriever",
    "add_args_for_retriever",
    "load_retrievers",
]
