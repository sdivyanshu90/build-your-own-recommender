# Testing strategy

Tests are deterministic and isolated under unit, integration, contract, property, regression,
security, E2E, failure-injection, concurrency, and performance categories. Unit tests cover strict
configuration, generation/cleaning, temporal leakage, train-only transformations, special tokens,
samplers, towers, normalization, loss, exact metrics/index, reranking, caching, paths, and manifests.
The integration test executes data through a live runtime; E2E adds the HTTP contract.

```bash
uv run pytest -m "not performance"
uv run pytest -m integration
uv run pytest -m e2e
uv run pytest -m security
uv run pytest -m performance
uv run pytest --cov --cov-branch --cov-report=html
```

Performance thresholds must be established on a named runner class; functional CI should not fail
on noisy microbenchmarks. No unit test calls an external API. Coverage targets are 90% line and 85%
branch, supplemented by behavioral and fault-oriented review.
