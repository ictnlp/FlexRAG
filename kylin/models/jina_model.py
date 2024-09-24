from dataclasses import dataclass

import requests
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
class JinaEncoderConfig(EncoderBaseConfig):
    model: str = "jina-embeddings-v3"
    base_url: str = "https://api.jina.ai/v1/embeddings"
    api_key: str = MISSING
    dimensions: int = 1024


@Encoders("jina", config_class=JinaEncoderConfig)
class JinaEncoder(EncoderBase):
    def __init__(self, cfg: JinaEncoderConfig):
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        self.base_url = cfg.base_url
        self._data_template = {
            "model": cfg.model,
            "task": "text-matching",
            "dimensions": cfg.dimensions,
            "late_chunking": False,
            "embedding_type": "float",
            "input": [],
        }
        return

    @TimeMeter("jina_encode")
    def encode(self, texts: list[str]) -> ndarray:
        data = self._data_template.copy()
        data["input"] = texts
        response = requests.post(self.base_url, headers=self.headers, json=data)
        response.raise_for_status()
        embeddings = [i["embedding"] for i in response.json()["data"]]
        return np.array(embeddings)

    @property
    def embedding_size(self) -> int:
        return self._data_template["dimension"]


@dataclass
class JinaRankerConfig(RankerConfig):
    model: str = "jina-reranker-v2-base-multilingual"
    base_url: str = "https://api.jina.ai/v1/rerank"
    api_key: str = MISSING


@Rankers("jina", config_class=JinaRankerConfig)
class JinaRanker(RankerBase):
    def __init__(self, cfg: JinaRankerConfig) -> None:
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        }
        self.base_url = cfg.base_url
        self._data_template = {
            "model": cfg.model,
            "query": "",
            "top_n": 0,
            "documents": [],
        }
        return

    @TimeMeter("jina_rank")
    def rank(self, query: str, candidates: list[str]) -> RankingResult:
        data = self._data_template.copy()
        data["query"] = query
        data["documents"] = candidates
        data["top_n"] = len(candidates)
        response = requests.post(self.base_url, json=data, headers=self.headers)
        response.raise_for_status()
        scores = [i["relevance_score"] for i in response.json()["results"]]
        ranking = np.argsort(scores)[::-1]
        return RankingResult(
            query=query,
            candidates=candidates,
            scores=scores,
            ranking=ranking.tolist(),
        )
