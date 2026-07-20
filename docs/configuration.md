# Configuration

Pydantic models reject unknown fields and invalid ranges. Precedence is model defaults, YAML, then
environment variables using `RECOMMENDER__SECTION__FIELD`. Environment values parse as JSON when
possible. For example:

```bash
RECOMMENDER__TRAINING__BATCH_SIZE=512 \
RECOMMENDER__SERVING__MAX_K=50 \
uv run recommender train --config configs/demo.yaml
```

Model and index metrics must match; default K cannot exceed maximum K; split fractions must leave a
test partition. Paths are configured rather than hard-coded. Secrets do not belong in YAML or
effective-configuration logs. Use a secret manager or mounted secret reference in production.
