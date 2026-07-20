# Negative sampling

Catalog-scale softmax is expensive, and logs rarely identify true dislikes. In-batch negatives
reuse other items in a minibatch. Uniform negatives cover the catalog, popularity-weighted
negatives resemble exposure, and ANN hard negatives focus on currently confusing items. Each has
sampling bias.

Samplers exclude the known positive and supplied known-positive set where candidates remain. Fixed
seeds make explicit sampling reproducible. Large batches supply more negatives but increase
duplicate positives and device memory. Hard-negative indexes must match the checkpoint and feature
version. Cross-batch memory and importance-sampling corrections are documented extensions; they
require stale-vector and probability correction policies.
