# Batch inference

`batch-recommend` reads Parquet record batches rather than all users, writes deterministic numbered
parts, and skips completed parts on restart. A checkpoint records the last part completed; the final
manifest checksums all parts and records model/index versions. Input requires `user_id`.

Use a unique output version per logical job. Retrying the same input, configuration, artifact
versions, and part size is idempotent. Do not change source row order during a restart. Production
orchestrators should write to staging, validate row counts/list bounds/version columns, then publish
an atomic completion marker. Run batch work separately from latency-sensitive serving.
