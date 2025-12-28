import os
from typing import Optional

import httpx
import numpy as np

from flexrag.common import TIME_METER, configure

from .ranker_base import RANKERS
from .remote_ranker_base import RemoteRankerBase, RemoteRankerBaseConfig


@configure
class VoyageRankerConfig(RemoteRankerBaseConfig):
    """Configuration for VoyageRanker.

    :param model: The model to use. Default is "rerank-2".
    :type model: str
    :param base_url: The base URL of the Voyage rerank API. Default is "https://api.voyageai.com/v1/rerank".
    :type base_url: str
    :param api_key: The API key for the Voyage rerank API.
        If not provided, it will use the environment variable `VOYAGE_API_KEY`.
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    """

    model: str = "rerank-2"
    base_url: str = "https://api.voyageai.com/v1/rerank"
    api_key: str = os.environ.get("VOYAGE_API_KEY", "EMPTY")
    proxy: Optional[str] = None


@RANKERS("voyage", config_class=VoyageRankerConfig)
class VoyageRanker(RemoteRankerBase):
    async def _create_client(self, config: VoyageRankerConfig):
        self._data_template = {"model": config.model}
        return httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            },
            proxy=config.proxy,
            base_url=config.base_url,
            follow_redirects=True,
        )

    @TIME_METER("ranker.voyage_rerank")
    async def _async_rank_impl(
        self, client, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        data = self._data_template.copy()
        data["query"] = query
        data["documents"] = candidates
        data["top_k"] = len(candidates)
        response = await client.post("", json=data)
        response.raise_for_status()
        results = response.json()["results"]
        scores = np.zeros(len(candidates))
        for res in results:
            scores[res["index"]] = res["relevance_score"]
        return None, scores
