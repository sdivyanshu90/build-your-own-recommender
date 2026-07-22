# ADR 0001: two-tower model for candidate retrieval

## Status

Accepted for the implemented reference architecture.

## Context

The platform must retrieve candidates from a catalog too large for a rich joint model to score in
full on every request. It must run locally without paid services while retaining a production vector
search boundary and cold-start metadata path.

## Decision

Use independent PyTorch user and item towers with a shared configurable embedding dimension,
normalized cosine/dot scoring, in-batch multi-positive softmax, offline item encoding, and a
pluggable exact/FAISS index.

```mermaid
flowchart LR
  UF[User/context features] --> U[User tower]
  IF[Item features] --> I[Item tower]
  I --> O[(Offline index)]
  U --> Q[Online query]
  Q --> O
  O --> C[Candidates for downstream ranking/policy]
```

## Consequences

Positive: item computation is amortized offline; exact search provides a correctness oracle; ANN is
replaceable; metadata and IDs can be fused; serving latency is bounded.

Negative: joint user-item cross features cannot influence retrieval directly; negative sampling and
exposure bias shape geometry; model and index must be promoted together; a downstream ranker remains
necessary for rich objectives.

## Alternatives considered

- popularity only: robust fallback but insufficient personalization;
- matrix factorization: efficient but less flexible feature fusion;
- full cross-encoder: expressive but unsuitable for full-catalog online scoring;
- managed vector database only: operational option, not a modeling architecture;
- lexical/graph retrieval: valuable complementary channels, not replacements for this educational
  neural retrieval objective.

