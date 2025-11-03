"""OpenTelemetry tracing setup."""
from typing import Optional

from config import settings

# Global tracer instance
_tracer: Optional[object] = None
_tracer_provider = None


def setup_tracing() -> None:
    """Set up OpenTelemetry tracing if enabled.

    Initializes OpenTelemetry tracer for FastAPI/Streamlit hooks.
    No-op if settings.ENABLE_TRACING is False.
    """
    global _tracer, _tracer_provider

    if not settings.ENABLE_TRACING:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        _tracer_provider = TracerProvider()
        trace.set_tracer_provider(_tracer_provider)

        # Console exporter for development
        console_exporter = ConsoleSpanExporter()
        span_processor = BatchSpanProcessor(console_exporter)
        _tracer_provider.add_span_processor(span_processor)

        # Get tracer for this module
        _tracer = trace.get_tracer(__name__)

        # In production, you would configure other exporters here
        # e.g., OTLP exporter for Jaeger, Zipkin, etc.
        # Example:
        # from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        # otlp_exporter = OTLPSpanExporter(endpoint="http://jaeger:4317")
        # span_processor = BatchSpanProcessor(otlp_exporter)
        # _tracer_provider.add_span_processor(span_processor)

    except ImportError:
        # OpenTelemetry not installed, tracing is no-op
        _tracer = None
        _tracer_provider = None


def get_tracer() -> Optional[object]:
    """Get OpenTelemetry tracer instance.

    Returns:
        Tracer instance if tracing is enabled, None otherwise.
        Returns a no-op tracer if OpenTelemetry is not available.

    Note:
        Returns None if settings.ENABLE_TRACING is False or if
        OpenTelemetry is not installed. Call setup_tracing() first
        to initialize tracing.
    """
    global _tracer

    if not settings.ENABLE_TRACING:
        return None

    if _tracer is None:
        # Try to get tracer, or return None if not initialized
        try:
            from opentelemetry import trace

            return trace.get_tracer(__name__)
        except ImportError:
            return None

    return _tracer


# Initialize tracing on module import if enabled
setup_tracing()

