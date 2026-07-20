# Vector search and index operations

The exact backend computes all inner products and is the correctness oracle. The FAISS backend uses
HNSW inner-product search with configurable graph degree, construction effort, and search effort.
Every change to those parameters must be evaluated for ANN recall and latency. IVF or product
quantization can be added behind the same interface. Cosine mode relies on tower-side L2
normalization. Dot mode preserves vector magnitude.

An index manifest records backend, metric, dimension, item count, model/embedding versions,
configuration, checksums, creation time, and self-retrieval validation. Full rebuild is safest for a
new model. Incremental inserts are appropriate only when the model and processor are unchanged;
deletions need tombstones or backend removal plus eligibility filtering. Periodic full rebuilds
remove fragmentation and accumulated tombstones.

Publish into an immutable version directory, validate it, load a green runtime alongside blue, then
switch traffic atomically. Roll back by selecting the prior compatible model/index bundle. Never
pair a new model with an old index merely because dimensions match.

OpenSearch, Elasticsearch, Milvus, Pinecone, Weaviate, Qdrant, and pgvector can implement the same
`search(queries, k)` boundary. Adapters must define consistency, filtering, deletion, tenancy,
authentication, timeout, retry, and score-normalization behavior.
