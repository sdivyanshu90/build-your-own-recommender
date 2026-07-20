import pandas as pd

from recommender.features.processor import UNKNOWN_TOKEN, FeatureProcessor


def _frames():
    users = pd.DataFrame(
        [
            {
                "user_id": "u1",
                "age_bucket": "25-34",
                "country": "IN",
                "language": "en",
                "subscription_tier": "free",
                "device_preference": "mobile",
                "account_age_days": 10,
                "preferred_categories": "books|music",
                "activity_score": 2.0,
            }
        ]
    )
    items = pd.DataFrame(
        [
            {
                "item_id": "i1",
                "category": "books",
                "subcategory": "books-0",
                "language": "en",
                "brand": "b",
                "price_bucket": "low",
                "price": 10.0,
                "popularity": 0.5,
                "freshness_days": 2,
                "available": True,
            }
        ]
    )
    events = pd.DataFrame([{"user_id": "u1", "item_id": "i1", "device": "mobile", "position": 1}])
    return users, items, events


def test_unknown_values_map_to_unknown_token(tiny_config) -> None:
    users, items, events = _frames()
    processor = FeatureProcessor.fit(users, items, events, tiny_config)
    assert (
        processor.encode("user.country", "XX")
        == processor.vocabularies["user.country"][UNKNOWN_TOKEN]
    )


def test_numerical_values_are_train_normalized(tiny_config) -> None:
    users, items, events = _frames()
    processor = FeatureProcessor.fit(users, items, events, tiny_config)
    transformed = processor.transform_users(users)
    assert transformed.loc[0, "user_account_age_days_z"] == 0


def test_processor_round_trip(tiny_config, tmp_path) -> None:
    users, items, events = _frames()
    processor = FeatureProcessor.fit(users, items, events, tiny_config)
    directory = tmp_path / "features"
    processor.save(directory, tiny_config, "dataset-v001")
    assert FeatureProcessor.load(directory) == processor
