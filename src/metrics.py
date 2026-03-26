"""Prometheus metrics for the IPL Prediction API."""

import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

logger = logging.getLogger(__name__)

# ── Metrics ──────────────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "ipl_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ipl_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_COUNT = Counter(
    "ipl_predictions_total",
    "Total predictions made",
    ["model"],
)

MODELS_LOADED = Gauge(
    "ipl_models_loaded",
    "Number of loaded XGBoost models",
)

DB_MATCHES = Gauge(
    "ipl_database_matches_total",
    "Total matches in DuckDB",
)

CACHE_HITS = Counter("ipl_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("ipl_cache_misses_total", "Redis cache misses")


# ── Middleware ────────────────────────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count and latency for every HTTP request."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start

        path = request.url.path
        method = request.method
        status = str(response.status_code)

        REQUEST_COUNT.labels(method=method, endpoint=path, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)

        return response


# ── Metrics Endpoint ─────────────────────────────────────────────────────────

async def metrics_endpoint():
    """Expose Prometheus metrics at /metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
