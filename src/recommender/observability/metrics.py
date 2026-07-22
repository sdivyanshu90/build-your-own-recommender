"""Low-cardinality Prometheus service metrics."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter("recommender_requests_total", "Requests", ["endpoint", "status"])
LATENCY = Histogram("recommender_request_duration_seconds", "Request latency", ["endpoint"])
FALLBACKS = Counter("recommender_fallbacks_total", "Fallback responses", ["reason"])
CACHE_HITS = Counter("recommender_cache_hits_total", "Recommendation cache hits")
RETURNED_ITEMS = Histogram("recommender_returned_items", "Number of returned items")
READY = Gauge("recommender_ready", "Whether compatible serving artifacts are loaded")
