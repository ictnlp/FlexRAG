import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from omegaconf import MISSING

from librarian.utils import TIME_METER

from .ranker import RankerBase, RankerConfig, RANKERS


@dataclass
class CohereRankerConfig(RankerConfig):
    model: str = "rerank-multilingual-v3.0"
    base_url: Optional[str] = None
    api_key: str = MISSING
    proxy: Optional[str] = None


@RANKERS("cohere", config_class=CohereRankerConfig)
class CohereRanker(RankerBase):
    def __init__(self, cfg: CohereRankerConfig) -> None:
        super().__init__(cfg)
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

    @TIME_METER("cohere_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.rerank(
            query=query,
            documents=candidates,
            model=self.model,
            top_n=len(candidates),
        )
        scores = [i.relevance_score for i in result.results]
        return None, scores

    @TIME_METER("cohere_rank")
    async def _async_rank(self, query: str, candidates: list[str]):
        result = await asyncio.create_task(
            asyncio.to_thread(
                self.client.rerank,
                query=query,
                documents=candidates,
                model=self.model,
                top_n=len(candidates),
            )
        )
        scores = [i.relevance_score for i in result.results]
        return None, scores
