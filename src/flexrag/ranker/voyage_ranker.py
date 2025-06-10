import asyncio
import os
from typing import Optional

import numpy as np

from flexrag.utils import TIME_METER, configure

from .ranker import RANKERS, RankerBase, RankerBaseConfig


@configure
class VoyageRankerConfig(RankerBaseConfig):
    """The configuration for the Voyage ranker.

    :param model: the model name of the ranker. Default is "rerank-2".
    :type model: str
    :param api_key: the API key for the Voyage ranker.
        If not provided, it will use the environment variable `VOYAGE_API_KEY`.
        Defaults to None.
    :type api_key: str
    :param timeout: the timeout for the request. Default is 3.0.
    :type timeout: float
    :param max_retries: the maximum number of retries. Default is 3.
    :type max_retries: int
    """

    model: str = "rerank-2"
    api_key: Optional[str] = None
    timeout: float = 3.0
    max_retries: int = 3


@RANKERS("voyage", config_class=VoyageRankerConfig)
class VoyageRanker(RankerBase):
    """VoyageRanker: The ranker based on the Voyage API."""

    def __init__(self, cfg: VoyageRankerConfig) -> None:
        super().__init__(cfg)
        from voyageai import Client

        api_key = cfg.api_key or os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for Voyage is not provided. "
                "Please set it in the configuration or as an environment variable 'VOYAGE_API_KEY'."
            )
        self.client = Client(
            api_key=api_key, max_retries=cfg.max_retries, timeout=cfg.timeout
        )
        self.model = cfg.model
        return

    @TIME_METER("voyage_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.rerank(
            query=query,
            documents=candidates,
            model=self.model,
            top_k=len(candidates),
        )
        scores = [i.relevance_score for i in result.results]
        return None, scores

    @TIME_METER("voyage_rank")
    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        result = await asyncio.create_task(
            asyncio.to_thread(
                self.client.rerank,
                query=query,
                documents=candidates,
                model=self.model,
                top_k=len(candidates),
            )
        )
        scores = [i.relevance_score for i in result.results]
        return None, scores
