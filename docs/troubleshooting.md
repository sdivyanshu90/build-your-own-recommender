# Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| Installation fails | Python is not 3.12 or lock is stale | Install Python 3.12 and run `uv sync --frozen --all-extras` |
| FAISS import fails | `ann` extra/platform wheel absent | `uv sync --extra ann`; use exact backend while diagnosing |
| Invalid configuration | Unknown key/range or model-index mismatch | Read the Pydantic error and compare `configs/demo.yaml` |
| Missing data | `generate-data` was not run or paths differ | Run `make generate-data`; inspect effective paths |
| Empty training batch | No event passes positive threshold | Inspect positive counts in dataset manifest and label policy |
| NaN loss | Bad numeric inputs, extreme temperature, unstable LR | Validate transformed Parquet, increase temperature, lower LR |
| Dimension mismatch | Model/embedding/index versions mixed | Inspect manifests and rebuild the downstream artifact chain |
| Corrupt artifact | Checksum differs | Quarantine it and restore/rebuild from trusted inputs |
| Readiness fails | Missing or incompatible runtime bundle | Run `validate-index` and `inspect-artifact` for every dependency |
| Empty recommendations | Filters/allow-list exclude all fallbacks | Check availability and request constraints; preserve an eligible fallback pool |
| Slow retrieval | Exact index/catalog too large or candidate K high | Benchmark FAISS, bound K, shard/vector-service as needed |
| Determinism test differs | Different Python/Torch/hardware or nondeterministic kernel | Match environment metadata and documented tolerance |
| Docker OOM | Per-worker model/index duplication | Reduce workers, memory-map/share index, or raise pod limit |
