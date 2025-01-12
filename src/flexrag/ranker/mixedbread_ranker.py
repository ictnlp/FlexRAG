import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx
import numpy as np
from omegaconf import MISSING

from flexrag.utils import TIME_METER

from .ranker import RankerBase, RankerBaseConfig, RANKERS


@dataclass
class MixedbreadRankerConfig(RankerBaseConfig):
    """The configuration for the Mixedbread ranker.

    :param model: the model name of the ranker. Default is "mxbai-rerank-large-v1".
    :type model: str
    :param api_key: the API key for the Mixedbread ranker. Required.
    :type api_key: str
    :param base_url: the base URL of the Mixedbread ranker. Default is None.
    :type base_url: Optional[str]
    :param proxy: the proxy for the request. Default is None.
    :type proxy: Optional[str]
    """

    model: str = "mxbai-rerank-large-v1"
    base_url: Optional[str] = None
    api_key: str = MISSING
    proxy: Optional[str] = None


@RANKERS("mixedbread", config_class=MixedbreadRankerConfig)
class MixedbreadRanker(RankerBase):
    """MixedbreadRanker: The ranker based on the Mixedbread API."""

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

    @TIME_METER("mixedbread_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.reranking(
            query=query,
            input=candidates,
            model=self.model,
            top_k=len(candidates),
        )
        scores = [i.score for i in result.data]
        return None, scores

    @TIME_METER("mixedbread_rank")
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
