import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import httpx
from omegaconf import MISSING

from flexrag.utils import TIME_METER

from .ranker import RankerBase, RankerBaseConfig, RANKERS


@dataclass
class JinaRankerConfig(RankerBaseConfig):
    """The configuration for the Jina ranker.

    :param model: the model name of the ranker. Default is "jina-reranker-v2-base-multilingual".
    :type model: str
    :param base_url: the base URL of the Jina ranker. Default is "https://api.jina.ai/v1/rerank".
    :type base_url: str
    :param api_key: the API key for the Jina ranker. Required.
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    """

    model: str = "jina-reranker-v2-base-multilingual"
    base_url: str = "https://api.jina.ai/v1/rerank"
    api_key: str = os.environ.get("JINA_API_KEY", MISSING)
    proxy: Optional[str] = None


@RANKERS("jina", config_class=JinaRankerConfig)
class JinaRanker(RankerBase):
    """JinaRanker: The ranker based on the Jina API."""

    def __init__(self, cfg: JinaRankerConfig) -> None:
        super().__init__(cfg)
        # prepare client
        self.client = httpx.Client(
            base_url=cfg.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            proxy=cfg.proxy,
            follow_redirects=True,
        )
        self.async_client = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {cfg.api_key}",
            },
            proxy=cfg.proxy,
            follow_redirects=True,
        )

        # prepare data template
        self.data_template = {
            "model": cfg.model,
            "query": "",
            "top_n": 0,
            "documents": [],
        }
        return

    @TIME_METER("jina_rank")
    def _rank(self, query: str, candidates: list[str]) -> tuple[np.ndarray, np.ndarray]:
        data = self.data_template.copy()
        data["query"] = query
        data["documents"] = candidates
        data["top_n"] = len(candidates)
        response = self.client.post("", json=data)
        response.raise_for_status()
        scores = [i["relevance_score"] for i in response.json()["results"]]
        return None, scores

    @TIME_METER("jina_rank")
    async def _async_rank(
        self, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        data = self.data_template.copy()
        data["query"] = query
        data["documents"] = candidates
        data["top_n"] = len(candidates)
        response = await self.async_client.post("", json=data)
        response.raise_for_status()
        scores = [i["relevance_score"] for i in response.json()["results"]]
        return None, scores
