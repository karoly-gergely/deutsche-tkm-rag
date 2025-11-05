"""
Performance metrics tracking for counters and latency measurements.
Provides in-memory registry for API endpoint exposure and supports
decorator-based timing instrumentation across the application.
"""

import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

# In-memory registry for demo
_metrics_registry: dict[str, Any] = {
    "counters": {},
    "timers": {},
}


class MetricsCollector:
    """Simple metrics collector for tracking operations."""

    def __init__(self):
        """Initialize metrics collector."""
        self.counts: dict[str, int] = {}
        self.timings: dict[str, list[float]] = {}

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric.

        Args:
            metric_name: Name of the metric.
            value: Value to increment by.
        """
        self.counts[metric_name] = self.counts.get(metric_name, 0) + value
        # Update registry
        _metrics_registry["counters"][metric_name] = self.counts[metric_name]

    def record_timing(self, metric_name: str, duration: float) -> None:
        """Record a timing metric.

        Args:
            metric_name: Name of the metric.
            duration: Duration in seconds.
        """
        if metric_name not in self.timings:
            self.timings[metric_name] = []
        self.timings[metric_name].append(duration)
        # Update registry
        if metric_name not in _metrics_registry["timers"]:
            _metrics_registry["timers"][metric_name] = []
        _metrics_registry["timers"][metric_name].append(duration)

    def get_count(self, metric_name: str) -> int:
        """Get count for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Current count value.
        """
        return self.counts.get(metric_name, 0)

    def get_avg_timing(self, metric_name: str) -> float | None:
        """Get average timing for a metric.

        Args:
            metric_name: Name of the metric.

        Returns:
            Average duration in seconds, or None if no data.
        """
        timings = self.timings.get(metric_name)
        if not timings:
            return None
        return sum(timings) / len(timings)

    def reset(self) -> None:
        """Reset all metrics."""
        self.counts.clear()
        self.timings.clear()
        _metrics_registry["counters"].clear()
        _metrics_registry["timers"].clear()


# Global metrics instance
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return _metrics


def get_metrics_registry() -> dict[str, Any]:
    """Get the in-memory metrics registry.

    Returns:
        Dictionary with 'counters' and 'timers' keys containing metric data.
    """
    return _metrics_registry.copy()


@contextmanager
def measure_latency(name: str):
    """Context manager to measure latency in milliseconds.

    Args:
        name: Name of the metric.

    Yields:
        Elapsed time in milliseconds (as float). The value is available
        after the context block completes.

    Example:
        with measure_latency("query_processing") as elapsed_ms:
            # ... do work ...
            result = process_query()
        # elapsed_ms() returns the elapsed time in milliseconds
        print(f"Took {elapsed_ms()} ms")
    """
    start = time.perf_counter()
    elapsed_ms: list[float] = [0.0]  # Use list for mutability in nested scope

    def get_elapsed():
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - start) * 1000.0

    try:
        yield get_elapsed
    finally:
        elapsed_seconds = time.perf_counter() - start
        elapsed_ms[0] = elapsed_seconds * 1000.0
        get_metrics().record_timing(name, elapsed_ms[0])


def track_timing(metric_name: str):
    """Decorator to track function execution time.

    Args:
        metric_name: Name of the metric to record.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with measure_latency(metric_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator
