import asyncio
import os
from typing import Optional

import httpx
import numpy as np

from flexrag.utils import TIME_METER, configure

from .ranker import RANKERS, RankerBase, RankerBaseConfig


@configure
class CohereRankerConfig(RankerBaseConfig):
    """The configuration for the Cohere ranker.

    :param model: the model name of the ranker. Default is "rerank-multilingual-v3.0".
    :type model: str
    :param base_url: the base URL of the Cohere ranker. Default is None.
    :type base_url: Optional[str]
    :param api_key: the API key for the Cohere ranker.
        If not provided, it will use the environment variable `COHERE_API_KEY`.
        Defaults to None.
    :type api_key: str
    :param proxy: the proxy for the request. Default is None.
    :type proxy: Optional[str]
    """

    model: str = "rerank-v3.5"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None


@RANKERS("cohere", config_class=CohereRankerConfig)
class CohereRanker(RankerBase):
    """CohereRanker: The ranker based on the Cohere API."""

    def __init__(self, cfg: CohereRankerConfig) -> None:
        super().__init__(cfg)
        from cohere import Client

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None

        api_key = cfg.api_key or httpx_client or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for Cohere is not provided. "
                "Please set it in the configuration or as an environment variable 'COHERE_API_KEY'."
            )
        self.client = Client(
            api_key=api_key, base_url=cfg.base_url, httpx_client=httpx_client
        )
        self.model = cfg.model
        return

    @TIME_METER("ranker.cohere_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.rerank(
            query=query,
            documents=candidates,
            model=self.model,
            top_n=len(candidates),
        )
        scores = [i.relevance_score for i in result.results]
        return None, scores

    @TIME_METER("ranker.cohere_rank")
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
