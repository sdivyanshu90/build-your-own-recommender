# Feature engineering

Categorical values use deterministic train-only vocabularies. Index 0 is padding and index 1 is
unknown. Values below minimum frequency map to unknown. Numerical fields use training mean and
population standard deviation; missing values impute to the training mean and therefore normalize
to zero. User category preferences use padded multi-hot token IDs and mean-pooled embeddings.
Sequences are bounded by configuration.

Fitting on validation, test, or the live catalog leaks future category presence and statistics.
The persisted processor is loaded by training, embedding export, evaluation, and serving. Its
manifest records schema and vocabulary hashes. Schema changes require a new feature version and
model retraining; incompatible versions fail rather than silently skewing inference.

Text embedding is an intentional extension point. Production options include a separately
versioned encoder or governed precomputed vectors; its version and dimension must enter the feature
and model contracts.
