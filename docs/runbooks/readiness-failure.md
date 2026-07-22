# Runbook: readiness failure after deployment

## Triage

```mermaid
flowchart TD
  R[Pod live but not ready] --> L[Inspect startup logs and /version]
  L --> A{Artifact path/read permission?}
  A -->|no| P[Fix mount, mode, UID, or delivery]
  A -->|yes| M{Manifest/checksum valid?}
  M -->|no| Q[Quarantine and restore trusted artifact]
  M -->|yes| C{Version/metric/dimension compatible?}
  C -->|no| B[Deploy complete matching bundle]
  C -->|yes| O{OOM/load timeout?}
  O -->|yes| K[Adjust resources/startup budget or reduce bundle]
  O -->|no| E[Escalate with diagnostic packet]
```

## Diagnostic commands

```bash
kubectl describe pod POD -n recommender
kubectl logs POD -n recommender --previous
kubectl logs POD -n recommender
kubectl get events -n recommender --sort-by=.lastTimestamp
uv run recommender inspect-artifact artifacts/models/model-v001 --config configs/production.yaml
uv run recommender validate-index --config configs/production.yaml
```

Run local artifact commands only against the exact bundle mounted in the failing revision. Preserve
the safe error and manifest; do not include secrets or raw user data in escalation.

