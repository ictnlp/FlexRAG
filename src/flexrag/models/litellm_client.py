from dataclasses import dataclass
from typing import Any

from aiohttp import ClientSession, TCPConnector
from litellm import AsyncHTTPHandler


@dataclass
class LiteLLMClient:
    """Provider-neutral request state and HTTP session ownership for LiteLLM."""

    model: str
    request_kwargs: dict[str, Any]
    owned_session: ClientSession | None = None

    @classmethod
    async def create(
        cls,
        *,
        provider: str,
        model_name: str,
        request_kwargs: dict[str, Any],
        max_concurrency: int,
    ) -> "LiteLLMClient":
        request_kwargs = dict(request_kwargs)
        owned_session = None
        if request_kwargs.get("shared_session") is None:
            owned_session = ClientSession(
                connector=TCPConnector(limit=max(1, max_concurrency))
            )
            request_kwargs["shared_session"] = owned_session
        return cls(
            model=f"{provider}/{model_name}",
            request_kwargs=request_kwargs,
            owned_session=owned_session,
        )

    async def close(self) -> None:
        if self.owned_session is not None:
            await self.owned_session.close()
        return


@dataclass
class LiteLLMRerankClient:
    """LiteLLM rerank state with an explicitly owned async HTTP handler."""

    model: str
    request_kwargs: dict[str, Any]
    owned_client: AsyncHTTPHandler | None = None

    @classmethod
    async def create(
        cls,
        *,
        provider: str,
        model_name: str,
        request_kwargs: dict[str, Any],
        timeout: float | None,
    ) -> "LiteLLMRerankClient":
        request_kwargs = dict(request_kwargs)
        owned_client = None
        if request_kwargs.get("client") is None:
            owned_client = AsyncHTTPHandler(timeout=timeout)
            request_kwargs["client"] = owned_client
        return cls(
            model=f"{provider}/{model_name}",
            request_kwargs=request_kwargs,
            owned_client=owned_client,
        )

    async def close(self) -> None:
        if self.owned_client is not None:
            await self.owned_client.close()
        return
