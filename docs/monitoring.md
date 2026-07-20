# Drift and operational monitoring

The offline comparator reports row-count ratios, missing-rate changes, numeric population stability
index, and unknown-category rates. Production monitoring should additionally compare embedding
norms, similarity scores, catalog coverage, popularity concentration, fallback rate, model/index
age, feature contract hashes, and online/offline transformations.

Data drift changes inputs; concept drift changes the input-label relationship; label drift changes
outcome prevalence; training-serving skew means implementations or source freshness differ;
operational degradation affects availability or latency. Retraining may address the first three but
not a corrupt index or failing cache.

Example triggers: unknown categories over 5%, PSI over 0.2, fallback rate doubled for 30 minutes,
coverage down 20%, or model/index beyond policy age. Thresholds must be calibrated from stable
history and routed through change-controlled alerts. Validate a candidate model and index before
promotion rather than automatically deploying every retrain.
