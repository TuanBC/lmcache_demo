"""
=============================================================================
Prometheus Metrics Module
=============================================================================

Prometheus metrics for monitoring cache efficiency and LLM performance.

METRICS EXPOSED:
----------------
- router_requests_total: Total number of requests by agent
- router_ttft_seconds: Histogram of Time to First Token by agent
- router_cache_hits_total: Cache hits inferred from TTFT
- router_prefix_mismatches_total: Prefix hash mismatches (cache busters)
=============================================================================
"""

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.responses import Response

# =============================================================================
# Prometheus Metrics Definitions
# =============================================================================

# Request counters
REQUEST_COUNTER = Counter(
    "router_requests_total",
    "Total number of requests processed",
    ["agent", "status"],
)

# TTFT histogram with buckets optimized for LLM inference
TTFT_HISTOGRAM = Histogram(
    "router_ttft_seconds",
    "Time to first token in seconds",
    ["agent"],
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 60.0],
)

# Cache efficiency metrics
CACHE_HIT_COUNTER = Counter(
    "router_cache_hits_total",
    "Total cache hits inferred from TTFT",
    ["agent"],
)

CACHE_MISS_COUNTER = Counter(
    "router_cache_misses_total",
    "Total cache misses inferred from TTFT",
    ["agent"],
)

PREFIX_MISMATCH_COUNTER = Counter(
    "router_prefix_mismatches_total",
    "Total prefix hash mismatches (cache busters)",
    ["agent"],
)

# Current state gauges
COLD_CACHE_BASELINE = Gauge(
    "router_cold_cache_baseline_seconds",
    "TTFT baseline from first (cold cache) request",
)

CACHE_HIT_RATE = Gauge(
    "router_cache_hit_rate",
    "Current inferred cache hit rate (0.0-1.0)",
)

PREFIX_TOKENS = Gauge(
    "router_prefix_tokens_estimated",
    "Estimated token count of the shared prefix",
)


# =============================================================================
# Metric Recording Functions
# =============================================================================


def record_request(agent: str, status: str = "success") -> None:
    """Record a request for an agent."""
    REQUEST_COUNTER.labels(agent=agent, status=status).inc()


def record_ttft(agent: str, ttft_seconds: float) -> None:
    """Record TTFT for an agent."""
    TTFT_HISTOGRAM.labels(agent=agent).observe(ttft_seconds)


def record_cache_hit(agent: str) -> None:
    """Record an inferred cache hit."""
    CACHE_HIT_COUNTER.labels(agent=agent).inc()


def record_cache_miss(agent: str) -> None:
    """Record an inferred cache miss."""
    CACHE_MISS_COUNTER.labels(agent=agent).inc()


def record_prefix_mismatch(agent: str) -> None:
    """Record a prefix mismatch (cache buster)."""
    PREFIX_MISMATCH_COUNTER.labels(agent=agent).inc()


def set_cold_cache_baseline(ttft: float) -> None:
    """Set the cold cache baseline TTFT."""
    COLD_CACHE_BASELINE.set(ttft)


def set_cache_hit_rate(rate: float) -> None:
    """Set the current cache hit rate."""
    CACHE_HIT_RATE.set(rate)


def set_prefix_tokens(tokens: int) -> None:
    """Set the estimated prefix token count."""
    PREFIX_TOKENS.set(tokens)


# =============================================================================
# Endpoint Handler
# =============================================================================


async def prometheus_metrics_endpoint() -> Response:
    """
    Generate Prometheus metrics in text format.

    Returns metrics in Prometheus exposition format for scraping.
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
