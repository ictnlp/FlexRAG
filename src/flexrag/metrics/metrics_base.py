from typing import Any, Protocol, TypeAlias, runtime_checkable

from flexrag.common import Register

MetricReturn: TypeAlias = tuple[dict[str, float], dict[str, Any]]
SimpleMetricReturn: TypeAlias = float | int | MetricReturn


@runtime_checkable
class MetricCallable(Protocol):
    def __call__(self, **kwargs: Any) -> SimpleMetricReturn: ...


METRICS = Register[MetricCallable]("metrics")
