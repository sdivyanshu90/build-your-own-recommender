# Online serving

The FastAPI lifecycle loads and verifies a feature processor, restricted model state, embedding
metadata, and index before readiness. Requests are bounded by payload bytes, `top_k`, batch size,
identifier lengths, and filter-list sizes. Correlation IDs are returned, and errors do not expose
stack traces.

Endpoints are `/health/live`, `/health/ready`, `/version`, `/metrics`, `/v1/model-info`,
`/v1/recommendations`, `/v1/similar-items`, `/v1/batch-recommendations`, and
`/v1/item-embedding`. Recommendation responses include request, model and index versions, retrieval
and final scores, ranks, fallback status, cache status, and latency stages.

Known users exclude their historical impressions/interactions by default. Unknown users with no
features receive globally eligible popularity/freshness fallback. Optional features permit a
cold-start tower vector. Filtering never bypasses availability. Candidate exhaustion fills from
eligible fallbacks and records the reason.

Terminate TLS and enforce identity, authorization, quotas, and distributed rate limiting at the
gateway. Keep application validation as defense in depth. Use multiple single-threaded workers only
after measuring duplicated model/index memory; process-shared FAISS or a vector service may be more
efficient for very large indexes.
