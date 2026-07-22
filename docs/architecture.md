# Architecture

The system separates offline mutation from online read-only serving. Offline jobs create immutable,
checksummed artifacts. Serving loads one compatible bundle and does not mutate model weights,
feature vocabularies, embeddings, or index state during requests.

## Context

```mermaid
C4Context
  title Two-tower recommendation platform context
  Person(user, "Product user", "Receives personalized candidates")
  Person(operator, "ML/platform operator", "Builds, validates, promotes, and rolls back artifacts")
  System(recsys, "Recommendation platform", "Offline model lifecycle and online candidate retrieval")
  System_Ext(events, "Event/entity sources", "Users, items, impressions, outcomes")
  System_Ext(consumer, "Ranking/product service", "Consumes candidate IDs and scores")
  System_Ext(obs, "Observability stack", "Metrics, logs, dashboards, alerts")
  Rel(user, recsys, "Requests recommendations", "HTTPS/JSON")
  Rel(events, recsys, "Supplies governed data", "Parquet in local implementation")
  Rel(operator, recsys, "Runs lifecycle and promotion commands")
  Rel(recsys, consumer, "Returns versioned candidates")
  Rel(recsys, obs, "Exports telemetry")
```

## Containers and ownership

```mermaid
flowchart TB
  CLI[Typer lifecycle CLI]
  DP[Data + feature pipeline]
  TR[PyTorch trainer]
  EV[Evaluator]
  IX[Embedding/index builder]
  AS[(Filesystem artifact store)]
  ML[(MLflow-compatible tracking)]
  API[FastAPI recommendation service]
  CA[(Memory/Redis cache)]
  PM[Prometheus]

  CLI --> DP --> AS
  CLI --> TR --> AS
  TR --> ML
  CLI --> EV
  EV --> AS
  CLI --> IX --> AS
  AS --> API
  API <--> CA
  API --> PM
```

| Container | Package ownership | Writes durable state? | Scaling model |
|---|---|---:|---|
| Lifecycle CLI | `recommender.cli` | Yes, through pipelines | Job/process per invocation |
| Data pipeline | `data`, `features` | Dataset and processor artifacts | Single-host local; distributed adapter extension |
| Trainer | `models`, `training`, `sampling` | Checkpoints, card, tracking metadata | CPU/single GPU implementation |
| Evaluator | `evaluation` | JSON/CSV/Markdown reports | Offline batch |
| Index builder | `embeddings`, `indexing` | Vectors and index artifacts | Offline batch |
| Online API | `serving`, `retrieval`, `reranking` | No model/index mutation | Replicated immutable processes |
| Control boundaries | `artifacts`, `security`, `monitoring` | Manifests/reports | Shared libraries |

## Offline data and model flow

```mermaid
flowchart LR
  R[(Raw Parquet)] --> V{Schema and value validation}
  V -->|findings| QR[Quality report]
  V --> C[Deterministic cleanup]
  C --> L[Event weighting and labels]
  L --> T[Global temporal split]
  T --> FT[Fit processor on train entities/events]
  FT --> X[Transform every split with frozen processor]
  X --> M[Train towers]
  M --> CK[Best/final weights + model card]
  CK --> E[Exact evaluation]
  CK --> EE[Encode eligible items]
  EE --> AI[Build exact or HNSW index]
  AI --> PV[Validate and publish version]
```

### Leakage boundary

The processor fits vocabularies and numerical statistics only from users/items observed in the
training event window. Validation and test transformations can map unseen categories to `<UNK>`,
but cannot expand the vocabulary or alter statistics. The model never trains on test positives.

## Online request flow

```mermaid
sequenceDiagram
  autonumber
  actor Client
  participant MW as Request middleware
  participant Cache
  participant RT as Runtime bundle
  participant UT as User tower
  participant ANN as Vector index
  participant Policy as Filters/reranker
  Client->>MW: POST /v1/recommendations
  MW->>MW: size + schema + K + correlation validation
  MW->>Cache: lookup bounded request key
  alt cache hit
    Cache-->>MW: immutable prior result
  else cache miss
    MW->>RT: user, context, exclusions, policies
    alt known or feature-supplied user
      RT->>UT: transformed feature batch
      UT-->>RT: normalized query vector
      RT->>ANN: search overfetch count
      ANN-->>RT: IDs and retrieval scores
    else unknown without usable features
      RT->>RT: construct eligible popularity/freshness fallback
    end
    RT->>Policy: availability, seen, allow/deny, category, freshness
    Policy-->>RT: deduplicated diverse top-K
    RT->>Cache: write with TTL
    RT-->>MW: candidates + versions + latency + fallback
  end
  MW-->>Client: safe structured response
```

## Runtime bundle and compatibility

```mermaid
classDiagram
  class RecommendationRuntime {
    +FeatureProcessor processor
    +TwoTowerModel model
    +VectorIndex index
    +DataFrame users
    +DataFrame items
    +Map seen
    +recommend()
    +similar_items()
    +item_embedding()
  }
  class ArtifactManifest {
    +artifact_type
    +version
    +config_hash
    +schema_hash
    +dependencies
    +files
    +require_dependency()
  }
  class VectorIndex {
    <<protocol>>
    +item_ids
    +dimension
    +search(query, k)
  }
  RecommendationRuntime --> ArtifactManifest : verifies
  RecommendationRuntime --> VectorIndex : read-only search
```

Startup loads and verifies feature, model, embedding, and index manifests before declaring
readiness. Files are checksummed before use. Model state is loaded with PyTorch `weights_only=True`.
String item IDs are fixed-width NumPy Unicode arrays and loaded with pickle disabled.

## Artifact state machine

```mermaid
stateDiagram-v2
  [*] --> Building
  Building --> Rejected: validation/checksum failure
  Building --> Validated: quality + compatibility pass
  Validated --> Published: atomic manifest/pointer update
  Published --> Loading: candidate deployment
  Loading --> Ready: startup checks pass
  Loading --> Rejected: mismatch/corruption
  Ready --> Retired: blue-green switch
  Retired --> Ready: rollback
```

Publication is downstream-only: a model cannot silently consume a different feature processor, and
an index cannot silently serve vectors from another model version.

## Failure boundaries

| Failure | Containment behavior | User-visible effect |
|---|---|---|
| Malformed raw data | Finding report and deterministic exclusion | Offline job may fail in strict mode |
| Empty cleaned dataset | Typed `DataQualityError` | No artifact published |
| Corrupt artifact | Checksum/manifest exception | Readiness remains false |
| Model-index mismatch | Dependency/metric/dimension rejection | Deployment does not enter service |
| Unknown user | Cold-feature encoding or fallback pool | Non-empty response when eligible items exist |
| Unknown item | Typed similar-item fallback reason | Safe response, no stack trace |
| Cache unavailable | Adapter boundary permits bypass policy | Higher latency, core retrieval remains possible |
| Over-filtering | Eligible fallback attempt | Fallback metadata or bounded short result |
| Request timeout | Safe structured error | No internal exception details |

## Scaling roadmap

```mermaid
flowchart TB
  L[Current: single-host Parquet + single model/index per process]
  L --> D[Distributed feature preparation: Spark/Beam]
  L --> T[Distributed training: DDP + cross-batch memory]
  L --> V[Managed/sharded vector service]
  V --> O[OpenSearch / Milvus / Qdrant / pgvector adapter]
  L --> FS[Online feature store with point-in-time correctness]
  L --> RK[Learned downstream ranker]
  RK --> EX[Experimentation + causal guardrails]
  L --> CP[Signed artifact registry and promotion controller]
```

These are explicit extensions, not hidden dependencies of the local workflow.

