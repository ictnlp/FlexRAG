from typing import Any

from flexrag.common.dataclasses import RetrievedContext


class RefinerInvocation:
    """Invocation semantics for managed refiners."""

    def __init__(self, runtime: Any) -> None:
        """Create a refiner invocation."""
        self.runtime = runtime
        return

    def refine(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Refine retrieved contexts."""
        return self.runtime.call("refine", contexts)
