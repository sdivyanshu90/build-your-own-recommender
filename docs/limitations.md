# Current limitations and scaling roadmap

The core pipeline is local pandas/PyArrow and assumes artifacts fit on one training host. FAISS uses
a flat inner-product index. Tracking configuration exists, while robust retry-buffered MLflow event
logging and OpenTelemetry export need deployment adapters. Redis/PostgreSQL are declared optional
boundaries but the default implementation is local/in-memory. Evaluation does not yet emit full
bootstrap confidence intervals or every requested business segment. Authentication and global rate
limiting belong at the deployment gateway.

Scaling path: replace local preparation with Spark/Beam behind the same schemas; use distributed
PyTorch with globally correct negatives; export sharded embeddings; evaluate IVF/HNSW/PQ against the
exact oracle; store signed artifacts in object storage; serve through sharded FAISS or a vector
database; add an online feature store and downstream ranker; deploy exploration and counterfactual
logging; automate governed retraining and canary promotion.
