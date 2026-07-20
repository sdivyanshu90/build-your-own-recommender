# Architecture

## Context and containers

```mermaid
flowchart TB
  Client --> API[Recommendation API]
  API --> Bundle[Immutable model/index bundle]
  API --> Metrics[Prometheus]
  Pipeline[Offline pipeline] --> Store[Versioned artifact store]
  Store --> Bundle
  Pipeline --> MLflow[MLflow-compatible tracking]
```

```mermaid
sequenceDiagram
  participant C as Client
  participant A as API
  participant U as User tower
  participant I as Vector index
  participant R as Rules
  C->>A: validated request + request ID
  A->>U: transformed user/context
  U-->>A: query vector
  A->>I: top candidate search
  I-->>A: IDs and retrieval scores
  A->>R: seen, availability, filters, MMR
  R-->>A: ordered eligible items
  A-->>C: versions, scores, fallback, latency
```

Model, feature, embedding, and index manifests form an explicit compatibility graph. Runtime
startup verifies checksums and dependency versions before readiness. Loaded tensors and indexes
are read-only during requests. Blue-green deployments load a second complete bundle, validate it,
then atomically replace the process reference; in-flight requests retain the old bundle.

Failure boundaries: invalid data is reported and quarantined; a corrupt artifact blocks
publication/readiness; unknown identities use fallbacks; cache failure should bypass cache;
candidate exhaustion invokes an eligible fallback. A separate worker or deployment is recommended
for batch inference so it cannot starve online serving.
