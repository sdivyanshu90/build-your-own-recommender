# Final implementation audit

Audit date: 2026-07-20. The checks below were executed in the provided workspace; generated data,
artifacts, reports, virtual environments, and coverage files are ignored by Git.

## Issues found and corrected

- Missing-value special tokens could create non-contiguous vocabulary indices and crash embedding
  lookup. Special markers are now reserved at indices 0 and 1, with a regression test.
- String item IDs were initially persisted as unsafe NumPy object arrays. They now use fixed-width
  Unicode arrays with pickle disabled.
- MLflow's maintained local path requires a SQL backend. Local tracking now uses SQLite with
  SQLAlchemy/Alembic; production can use the Compose MLflow server.
- The first lock selected vulnerable PyArrow and pytest releases. Constraints and lock were updated;
  the final `pip-audit` reported no known vulnerabilities.
- The initial Linux PyTorch lock included unused CUDA libraries. The default lock now selects the
  official CPU wheel index; accelerator builds require a separate reviewed lock.
- The first container virtual environment contained builder-path shebangs. Builder and runtime now
  share `/app`, and the installed entry point executes in the runtime image.
- Atomic JSON publication inherited mode `0600`, preventing the non-root container from reading
  host-built manifests. Publication now sets `0644`, covered by a test.
- Docker dependency installation invalidated on every source edit. Dependency and project install
  layers are now separated with a BuildKit cache mount.
- FAISS initially exercised only flat exact persistence. The FAISS backend now uses configurable
  HNSW approximate inner-product search, while the exact backend remains the correctness oracle.
- Training configuration accepted explicit negative samplers without using them and ignored resume
  paths. Non-in-batch modes now fail clearly, while valid local `.pt` checkpoints are loaded with
  PyTorch's weights-only loader and recorded in training metadata.

## Executed verification

| Check | Result |
|---|---|
| Locked Python environment | CPython 3.12.13, `uv sync --frozen --all-extras` succeeded |
| Formatting and lint | Ruff format check and lint passed |
| Static typing | Strict mypy passed for 64 source files |
| Automated tests | 55 tests passed after the final source change |
| Coverage | 96.28% statements and 85.62% branches, above the stated targets |
| Documentation | `mkdocs build --strict` passed |
| Static security | Bandit and detect-secrets passed |
| Dependency audit | `pip-audit`: no known vulnerabilities in third-party packages |
| Demo | `make demo` completed; training loss decreased across its three configured epochs |
| Docker build | CPU-only multi-stage image built successfully |
| Container security | Configured UID/GID `10001:10001`, read-only root, tmpfs `/tmp`, read-only artifacts |
| Container smoke | `/health/ready` and `/v1/recommendations` succeeded with model/index version metadata |

CI repeats these checks from the lockfile on Python 3.12; they should be rerun after any source edit.

## Remaining limitations

- The native trainer's optimized default is in-batch softmax. Uniform, popularity-weighted, and
  ANN hard-negative samplers are implemented and tested as pluggable components, but a distributed
  sampled-softmax training loop and cross-batch memory are future extensions.
- Checkpoint resume restores model weights safely, but not optimizer/scheduler state or the prior
  epoch counter. Full fault-tolerant continuation remains an extension.
- Local preparation uses pandas/PyArrow on one host. Spark/Beam adapters and distributed training
  are interfaces/roadmap work, not implemented claims.
- HNSW parameters are defaults, not capacity recommendations. Production must measure ANN recall,
  latency, memory, and update behavior on representative catalogs.
- The lightweight policy layer is not a learned downstream ranker. Business deployment needs a
  governed ranking model and controlled online experiments.
- Redis and PostgreSQL adapters are optional and injected; the default runtime uses in-memory/local
  state. Authentication, global rate limiting, TLS, and manifest signing remain platform controls.
- Raw Kubernetes manifests were statically reviewed and YAML-parsed; no live cluster deployment was
  available in this environment.
