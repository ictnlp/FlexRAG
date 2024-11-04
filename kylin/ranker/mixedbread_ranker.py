import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from omegaconf import MISSING

from kylin.utils import TimeMeter

from .ranker import RankerBase, RankerConfig, Rankers


@dataclass
class MixedbreadRankerConfig(RankerConfig):
    model: str = "mxbai-rerank-large-v1"
    base_url: Optional[str] = None
    api_key: str = MISSING
    proxy: Optional[str] = None


@Rankers("mixedbread", config_class=MixedbreadRankerConfig)
class MixedbreadRanker(RankerBase):
    def __init__(self, cfg: MixedbreadRankerConfig) -> None:
        super().__init__(cfg)
        from mixedbread_ai.client import MixedbreadAI

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None
        self.client = MixedbreadAI(
            api_key=cfg.api_key, base_url=cfg.base_url, httpx_client=httpx_client
        )
        self.model = cfg.model
        return

    @TimeMeter("mixedbread_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.reranking(
            query=query,
            input=candidates,
            model=self.model,
            top_k=len(candidates),
        )
        scores = [i.score for i in result.data]
        return None, scores

    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        result = await asyncio.create_task(
            asyncio.to_thread(
                self.client.reranking,
                query=query,
                input=candidates,
                model=self.model,
                top_k=len(candidates),
            )
        )
        scores = [i.score for i in result.data]
        return None, scores
