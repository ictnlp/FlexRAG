from .elastic_retriever import ElasticRetriever, ElasticRetrieverConfig
from .flex_retriever import FlexRetriever, FlexRetrieverConfig
from .hyde_retriever import HydeRetriever, HydeRetrieverConfig
from .retriever_base import (
    RETRIEVERS,
    EditableRetriever,
    EditableRetrieverConfig,
    LocalRetriever,
    LocalRetrieverConfig,
    RetrieverBase,
    RetrieverBaseConfig,
)
from .typesense_retriever import TypesenseRetriever, TypesenseRetrieverConfig
from .web_retrievers import (
    SimpleWebRetriever,
    SimpleWebRetrieverConfig,
    WikipediaRetriever,
    WikipediaRetrieverConfig,
)

RetrieverConfig = RETRIEVERS.make_config(config_name="RetrieverConfig", default=None)


__all__ = [
    "ElasticRetriever",
    "ElasticRetrieverConfig",
    "FlexRetriever",
    "FlexRetrieverConfig",
    "HydeRetriever",
    "HydeRetrieverConfig",
    "RETRIEVERS",
    "EditableRetriever",
    "EditableRetrieverConfig",
    "LocalRetriever",
    "LocalRetrieverConfig",
    "RetrieverBase",
    "RetrieverBaseConfig",
    "TypesenseRetriever",
    "TypesenseRetrieverConfig",
    "SimpleWebRetriever",
    "SimpleWebRetrieverConfig",
    "WikipediaRetriever",
    "WikipediaRetrieverConfig",
    "RetrieverConfig",
]
