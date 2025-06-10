import asyncio
import os
from typing import Optional

import httpx
import numpy as np

from flexrag.utils import TIME_METER, configure

from .ranker import RANKERS, RankerBase, RankerBaseConfig


@configure
class MixedbreadRankerConfig(RankerBaseConfig):
    """The configuration for the Mixedbread ranker.

    :param model: the model name of the ranker. Default is "mxbai-rerank-base-v2".
    :type model: str
    :param api_key: the API key for the Mixedbread ranker.
        If not provided, it will use the environment variable `MIXEDBREAD_API_KEY`.
        Defaults to None.
    :type api_key: str
    :param base_url: the base URL of the Mixedbread ranker. Default is None.
    :type base_url: Optional[str]
    :param proxy: the proxy for the request. Default is None.
    :type proxy: Optional[str]
    """

    model: str = "mxbai-rerank-base-v2"
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    proxy: Optional[str] = None


@RANKERS("mixedbread", config_class=MixedbreadRankerConfig)
class MixedbreadRanker(RankerBase):
    """MixedbreadRanker: The ranker based on the Mixedbread API."""

    def __init__(self, cfg: MixedbreadRankerConfig) -> None:
        super().__init__(cfg)
        from mixedbread import Mixedbread

        if cfg.proxy is not None:
            httpx_client = httpx.Client(proxies=cfg.proxy)
        else:
            httpx_client = None

        api_key = cfg.api_key or httpx_client or os.getenv("MIXEDBREAD_API_KEY")
        if not api_key:
            raise ValueError(
                "API key for Mixedbread is not provided. "
                "Please set it in the configuration or as an environment variable 'MIXEDBREAD_API_KEY'."
            )
        self.client = Mixedbread(
            api_key=api_key, base_url=cfg.base_url, http_client=httpx_client
        )
        self.model = cfg.model
        return

    @TIME_METER("mixedbread_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        result = self.client.rerank(
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
                self.client.rerank,
                query=query,
                input=candidates,
                model=self.model,
                top_k=len(candidates),
            )
        )
        scores = [i.score for i in result.data]
        return None, scores
