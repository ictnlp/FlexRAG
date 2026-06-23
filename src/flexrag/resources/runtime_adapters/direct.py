import inspect
from typing import Any


class DirectRuntimeAdapter:
    """Direct runtime adapter for main-process raw resources.

    The adapter constructs a raw resource implementation and forwards public
    calls to it without adding execution isolation or remote runtime policies.
    It is intended for lightweight resources whose heavy dependencies, if any,
    are injected as separately managed resources.
    """

    impl_cls: type[Any] | None = None

    def __init__(
        self,
        config: Any,
        impl_cls: type[Any] | None = None,
        **dependencies: Any,
    ) -> None:
        """Create a direct runtime adapter.

        :param config: Configuration passed to the raw implementation.
        :param impl_cls: Optional raw implementation class. When omitted,
            subclasses must set ``impl_cls``.
        :param dependencies: Externally managed resources injected into the raw
            implementation constructor.
        :raises ValueError: If no implementation class is configured.
        """
        if impl_cls is not None:
            self.impl_cls = impl_cls
        if self.impl_cls is None:
            raise ValueError(f"{self.__class__.__name__}.impl_cls must be configured.")
        self._resource = self.impl_cls(config, **dependencies)
        return

    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes and methods to the wrapped raw resource."""
        return getattr(self._resource, name)

    def getattr(self, name: str) -> Any:
        """Return an attribute from the wrapped raw resource.

        :param name: Attribute name.
        :return: Attribute value.
        """
        return getattr(self._resource, name)

    async def agetattr(self, name: str) -> Any:
        """Return an attribute from the wrapped raw resource asynchronously.

        :param name: Attribute name.
        :return: Attribute value.
        """
        return self.getattr(name)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the wrapped raw resource.

        :param method: Method name.
        :param args: Positional arguments forwarded to the method.
        :param kwargs: Keyword arguments forwarded to the method.
        :return: Method return value.
        """
        return getattr(self._resource, method)(*args, **kwargs)

    async def acall(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Call a method on the wrapped raw resource asynchronously.

        Awaitable return values are awaited before they are returned.

        :param method: Method name.
        :param args: Positional arguments forwarded to the method.
        :param kwargs: Keyword arguments forwarded to the method.
        :return: Method return value.
        """
        result = self.call(method, *args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    def close(self) -> None:
        """Close the wrapped resource when it exposes a synchronous close hook."""
        close = getattr(self._resource, "close", None)
        if callable(close):
            close()
        return

    async def aclose(self) -> None:
        """Close the wrapped resource, preferring an async close hook when present."""
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
