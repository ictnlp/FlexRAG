import inspect
from typing import Any

from flexrag.common.dataclasses import RetrievedContext
from flexrag.processors.refiners.refiner_base import RefinerProtocol


class RefinerRuntimeAdapter:
    """Direct runtime adapter for main-process raw refiners.

    The adapter constructs a raw refiner implementation and forwards
    ``refine`` calls to it. Heavy execution, if any, is expected to live in
    injected encoder or generator resources rather than in the refiner itself.
    """

    impl_cls: type[RefinerProtocol] | None = None

    def __init__(
        self,
        config: Any,
        impl_cls: type[RefinerProtocol] | None = None,
        **dependencies: Any,
    ) -> None:
        """Create a direct refiner runtime adapter.

        :param config: Configuration passed to the raw refiner implementation.
        :param impl_cls: Optional raw refiner implementation class. When
            omitted, subclasses must set ``impl_cls``.
        :param dependencies: Externally managed resources injected into the
            raw refiner constructor.
        :raises ValueError: If no implementation class is configured.
        """
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        self._resource = self.impl_cls(config, **dependencies)
        return

    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Refine retrieved contexts.

        :param contexts: Retrieved contexts to refine.
        :return: Refined contexts.
        """
        return self._resource.refine(contexts)

    def close(self) -> None:
        """Close the wrapped refiner when it exposes a synchronous close hook."""
        close = getattr(self._resource, "close", None)
        if callable(close):
            close()
        return

    async def aclose(self) -> None:
        """Close the wrapped refiner, preferring an async close hook when present."""
        aclose = getattr(self._resource, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                await result
            return

        close = getattr(self._resource, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        return
