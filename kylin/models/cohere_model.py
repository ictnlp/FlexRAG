from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from numpy import ndarray
from omegaconf import MISSING

from kylin.utils import TimeMeter

from .model_base import (
    EncoderBase,
    EncoderBaseConfig,
    RankerBase,
    RankerConfig,
    RankingResult,
    Encoders,
    Rankers,
)


@dataclass
class CohereEncoderConfig(EncoderBaseConfig):
    model: str = "embed-multilingual-v3.0"
    input_type: str = "search_document"
    base_url: Optional[str] = None
    api_key: str = MISSING
    proxy: Optional[str] = None


@Encoders("cohere", config_class=CohereEncoderConfig)
class CohereEncoder(EncoderBase):
    def __init__(self, cfg: CohereEncoderConfig):
        from cohere import Client

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None
        self.client = Client(
            api_key=cfg.api_key, base_url=cfg.base_url, httpx_client=httpx_client
        )
        self.model = cfg.model
        self.input_type = cfg.input_type
        return

    @TimeMeter("cohere_encode")
    def encode(self, texts: list[str]) -> ndarray:
        r = self.client.embed(
            texts=texts,
            model=self.model,
            input_type=self.input_type,
            embedding_types=["float"],
        )
        embeddings = r.embeddings.float
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        return self._data_template["dimension"]


@dataclass
class CohereRankerConfig(RankerConfig):
    model: str = "rerank-multilingual-v3.0"
    base_url: Optional[str] = None
    api_key: str = MISSING
    proxy: Optional[str] = None


@Rankers("cohere", config_class=CohereRankerConfig)
class CohereRanker(RankerBase):
    def __init__(self, cfg: CohereRankerConfig) -> None:
        from cohere import Client

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None
        self.client = Client(
            api_key=cfg.api_key, base_url=cfg.base_url, httpx_client=httpx_client
        )
        self.model = cfg.model
        return

    @TimeMeter("cohere_rank")
    def rank(self, query: str, candidates: list[str]) -> RankingResult:
        result = self.client.rerank(
            query=query,
            documents=candidates,
            model=self.model,
            top_n=len(candidates),
        )
        scores = [i.relevance_score for i in result.results]
        ranking = np.argsort(scores)[::-1]
        return RankingResult(
            query=query,
            candidates=candidates,
            scores=scores,
            ranking=ranking.tolist(),
        )
