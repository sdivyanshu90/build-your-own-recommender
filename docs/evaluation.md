# Offline evaluation

For user $u$, Recall@K is retrieved relevant items divided by all relevant test items. Precision@K
uses K as denominator. Hit Rate is one if any relevant item appears. MRR is reciprocal rank of the
first hit. NDCG discounts hits logarithmically and normalizes by the ideal list. MAP averages
precision at hit positions.

Evaluation truth contains only positive test events. Train and validation interactions are excluded
from candidates to prevent metric inflation. Reports also include catalog/user coverage, a
self-information novelty proxy, category pairwise diversity, retrieval latency, and popularity
baseline. Empty relevance sets contribute no user row; short result lists still use K for precision.

The report segments maximum-K user metrics by new/existing lifecycle, sparse/active history,
geography, device, and subscription tier. Seeded bootstrap intervals cover Recall, NDCG, and MRR.
Item head share and mean recommended popularity expose concentration. Production reporting should
enforce minimum cohort sizes to protect privacy and add business-specific item-age/head-tail cuts.
ANN validation compares the FAISS HNSW index with exact/self-retrieval checks.

Historical exposure, selection, position, delayed labels, and feedback loops limit offline validity.
Use online experimentation to establish causal product impact.
