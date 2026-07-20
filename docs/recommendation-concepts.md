# Recommendation-system fundamentals

Recommendation systems solve several different problems. Collaborative filtering learns from
overlapping behavior; content systems match attributes; matrix factorization represents a sparse
interaction matrix with latent factors; neural retrieval generalizes that idea to arbitrary typed
features. Explicit feedback states preference directly, such as a rating. Implicit feedback—an
impression, click, or purchase—mixes preference with exposure and interface effects.

A large production stack commonly has four stages:

1. **Candidate generation** retrieves hundreds or thousands of plausible items from millions.
2. **Ranking** uses richer user-item cross features to estimate objectives such as engagement.
3. **Re-ranking** applies diversity, freshness, constraints, and multi-objective trade-offs.
4. **Filtering** enforces availability, policy, rights, deletion, and already-consumed rules.

This project completely implements stage one and a small boundary demonstration of stages three
and four. It does not claim to replace a feature-rich production ranker.

Historical logs are selected data. Items must be exposed before users can interact, so treating
all absent interactions as dislikes creates exposure bias. Position changes interaction
probability. Popular items accumulate feedback faster. Model recommendations change future data,
creating feedback loops. Delayed purchases and ratings make labels incomplete near a split
boundary. Random splitting can place later user intent or item state in training, inflating
offline results; this project splits globally by event time.

Exploration deliberately gathers information; exploitation chooses the current best estimate.
Offline replay cannot fully measure this counterfactual. Launch decisions require guarded online
experiments with product, safety, and fairness metrics.
