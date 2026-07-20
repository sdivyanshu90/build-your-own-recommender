# Deployment

The Docker build resolves the locked Python 3.12 environment in a builder and copies only the
virtual environment and configuration to a slim non-root runtime. Build and scan it before use:

```bash
docker build -t two-tower-recommender:local .
docker run --rm --read-only --tmpfs /tmp -p 8000:8000 \
  -v "$PWD/artifacts:/app/artifacts:ro" two-tower-recommender:local
```

Kubernetes manifests include service account, ConfigMap, secret template, probes, requests/limits,
rolling update, HPA, disruption budget, security context, ingress, network policy, and PVC.

Artifact delivery choices: baking gives immutable/simple startup but large images; an init container
downloads once but delays readiness; object-store download in application couples serving to
credentials/network; a sidecar can synchronize versions but needs atomic handoff; mounted volumes
are simple but depend on storage consistency. The sample uses a read-only mounted volume. Promote a
complete signed model/index bundle and keep the previous version for rollback.

The default lock selects PyTorch's official CPU wheel index to keep local and serving images free of
unused CUDA libraries. GPU training should use a separately reviewed accelerator lock/image whose
CUDA runtime matches the cluster drivers; do not silently change the serving lock.
