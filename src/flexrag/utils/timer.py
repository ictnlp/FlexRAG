import inspect
from collections import defaultdict
from time import perf_counter

import numpy as np


class _TimeMeter:
    def __init__(self):
        self.timers = defaultdict(list)
        return

    def __call__(self, *timer_names: str):
        def time_it(func):
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = func(*args, **kwargs)
                end_time = perf_counter()
                self.timers[timer_names].append(end_time - start_time)
                return result

            async def async_wrapper(*args, **kwargs):
                start_time = perf_counter()
                result = await func(*args, **kwargs)
                end_time = perf_counter()
                self.timers[timer_names].append(end_time - start_time)
                return result

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return time_it

    @property
    def statistics(self) -> list[dict[str, float]]:
        # TODO: support multiple process
        statistics = []
        for k, v in self.timers.items():
            v = list(v)
            statistics.append(
                {
                    "name": k,
                    "calls": len(v),
                    "average call time": np.mean(v),
                    "total time": np.sum(v),
                }
            )
        return statistics

    @property
    def details(self) -> dict:
        return {k: v for k, v in self.timers.items()}


TIME_METER = _TimeMeter()
