import os
from typing import Optional

import httpx
import numpy as np

from flexrag.common import TIME_METER, configure

from .ranker_base import RANKERS
from .remote_ranker_base import RemoteRankerBase, RemoteRankerBaseConfig


@configure
class CohereRankerConfig(RemoteRankerBaseConfig):
    """Configuration for CohereRanker.

    :param model: The model to use. Default is "rerank-v3.5".
    :type model: str
    :param base_url: The base URL of the Cohere rerank API. Default is "https://api.cohere.com/v1/rerank".
    :type base_url: str
    :param api_key: The API key for the Cohere rerank API.
        If not provided, it will use the environment variable `COHERE_API_KEY`.
    :type api_key: str
    :param proxy: The proxy to use. Defaults to None.
    :type proxy: Optional[str]
    """

    model: str = "rerank-v3.5"
    base_url: str = "https://api.cohere.com/v1/rerank"
    api_key: str = os.environ.get("COHERE_API_KEY", "EMPTY")
    proxy: Optional[str] = None


@RANKERS("cohere", config_class=CohereRankerConfig)
class CohereRanker(RemoteRankerBase):
    async def _create_client(self, config: CohereRankerConfig):
        self._data_template = {"model": config.model}
        return httpx.AsyncClient(
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
                "X-Client-Name": "FlexRAG",
            },
            proxy=config.proxy,
            base_url=config.base_url,
            follow_redirects=True,
        )

    @TIME_METER("ranker.cohere_rerank")
    async def _async_rank_impl(
        self, client, query: str, candidates: list[str]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        data = self._data_template.copy()
        data["query"] = query
        data["documents"] = candidates
        data["top_n"] = len(candidates)
        data["return_documents"] = False
        response = await client.post("", json=data)
        response.raise_for_status()
        results = response.json()["results"]
        scores = np.zeros(len(candidates))
        for res in results:
            scores[res["index"]] = res["relevance_score"]
        return None, scores
