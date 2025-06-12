import inspect
import resource
from collections import defaultdict
from copy import deepcopy
from time import perf_counter

import numpy as np


class _ResourceMeter:
    def __init__(self):
        self.wall_clock_timers = defaultdict(list)
        self.user_cpu_timers = defaultdict(list)
        self.system_cpu_timers = defaultdict(list)
        self.memory_usage = defaultdict(list)
        return

    def reset(self):
        """Reset all timers and memory usage."""
        self.wall_clock_timers.clear()
        self.user_cpu_timers.clear()
        self.system_cpu_timers.clear()
        self.memory_usage.clear()
        return

    def __call__(self, *timer_names: str):
        def time_it(func):
            def wrapper(*args, **kwargs):
                start_time = perf_counter()
                start_usage = resource.getrusage(resource.RUSAGE_SELF)
                result = func(*args, **kwargs)
                end_usage = resource.getrusage(resource.RUSAGE_SELF)
                end_time = perf_counter()
                self.wall_clock_timers[timer_names].append(end_time - start_time)
                self.user_cpu_timers[timer_names].append(
                    end_usage.ru_utime - start_usage.ru_utime
                )
                self.system_cpu_timers[timer_names].append(
                    end_usage.ru_stime - start_usage.ru_stime
                )
                self.memory_usage[timer_names].append(end_usage.ru_maxrss)
                return result

            async def async_wrapper(*args, **kwargs):
                start_time = perf_counter()
                start_usage = resource.getrusage(resource.RUSAGE_SELF)
                result = await func(*args, **kwargs)
                end_usage = resource.getrusage(resource.RUSAGE_SELF)
                if inspect.isawaitable(result):
                    result = await result
                else:
                    result = result
                end_time = perf_counter()
                self.wall_clock_timers[timer_names].append(end_time - start_time)
                self.user_cpu_timers[timer_names].append(
                    end_usage.ru_utime - start_usage.ru_utime
                )
                self.system_cpu_timers[timer_names].append(
                    end_usage.ru_stime - start_usage.ru_stime
                )
                self.memory_usage[timer_names].append(end_usage.ru_maxrss)
                return result

            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return wrapper

        return time_it

    @property
    def statistics(self) -> list[dict[str, float]]:
        # TODO: support multiple process
        statistics = []
        for timer_name in self.wall_clock_timers.keys():
            wall_clock_times = self.wall_clock_timers[timer_name]
            user_cpu_times = self.user_cpu_timers[timer_name]
            system_cpu_times = self.system_cpu_timers[timer_name]
            statistics.append(
                {
                    "name": timer_name,
                    "calls": len(wall_clock_times),
                    "average wall clock time": np.mean(wall_clock_times),
                    "total wall clock time": np.sum(wall_clock_times),
                    "average user cpu time": np.mean(user_cpu_times),
                    "total user cpu time": np.sum(user_cpu_times),
                    "average system cpu time": np.mean(system_cpu_times),
                    "total system cpu time": np.sum(system_cpu_times),
                    "maximum memory usage (MB)": np.max(self.memory_usage[timer_name])
                    / 1024,
                    "average memory usage (MB)": np.mean(self.memory_usage[timer_name])
                    / 1024,
                }
            )
        return statistics

    @property
    def details(self) -> dict:
        return {
            "wall_clock_timers": deepcopy(self.wall_clock_timers),
            "user_cpu_timers": deepcopy(self.user_cpu_timers),
            "system_cpu_timers": deepcopy(self.system_cpu_timers),
            "memory_usage": deepcopy(self.memory_usage),
        }


TIME_METER = _ResourceMeter()
