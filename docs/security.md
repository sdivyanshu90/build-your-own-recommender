# Security and privacy

## Threat model

| Threat | Controls in this repository | Required deployment control |
|---|---|---|
| Malicious/oversized payload | Strict Pydantic schemas, byte/list/K bounds, safe errors | Gateway WAF and distributed quotas |
| Denial of service | Bounded search and batch sizes, time budget configuration | Autoscaling, concurrency limits, circuit breakers |
| Enumeration/model extraction | Limited explanation metadata | Authentication, per-principal quotas, anomaly detection |
| Membership inference | No raw training evidence in responses | Privacy review, aggregation, optional DP techniques |
| Poisoned events | Validation, immutable manifests, quality reports | Source authorization and anomaly/quarantine workflow |
| Artifact replacement | Checksums and compatibility checks | Signed manifests, trusted registry, least-privilege promotion |
| Unsafe serialization | Restricted weights load; NumPy pickle disabled | Only load authenticated artifacts |
| Dependency compromise | Lockfile, Dependabot, audit/Bandit/secret checks | Provenance, signed images, SBOM and admission policy |
| Sensitive logging | Structured safe fields and no raw features | Central redaction tests and retention controls |

The container uses a non-root UID, drops Linux capabilities, prevents privilege escalation, and
supports a read-only root filesystem. Kubernetes disables service-account token mounting and uses a
default-deny-style network policy. TLS terminates at ingress. CORS is empty by default.

Artifact inspection confines user-provided paths to the configured artifact root. Checksums are not
a substitute for signature verification. Never load an artifact uploaded by an untrusted API user.
Do not commit `.env`; rotate any credential disclosed in chat, logs, history, or issue trackers.

Deletion workflows must purge source and derived identity/history records, caches, batch outputs,
and future datasets. Define retention by data class and legal basis. Existing trained parameters may
require expedited retraining. Salted user hashes require a secret deployment salt and rotation plan.
