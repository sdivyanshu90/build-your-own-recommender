from recommender.features.processor import PAD_TOKEN, UNKNOWN_TOKEN, FeatureProcessor


def test_missing_value_tokens_do_not_create_sparse_embedding_indices() -> None:
    vocabulary = FeatureProcessor._vocabulary([UNKNOWN_TOKEN, "a", PAD_TOKEN], 1)
    assert vocabulary[PAD_TOKEN] == 0
    assert vocabulary[UNKNOWN_TOKEN] == 1
    assert max(vocabulary.values()) == len(vocabulary) - 1
