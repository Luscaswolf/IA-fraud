"""Testes da engenharia de features."""
from src.features import FEATURE_COLS, build_features, haversine


def test_haversine_known_distance():
    # Sao Paulo -> Rio ~ 360 km
    d = haversine(-23.55, -46.63, -22.90, -43.20)
    assert 330 < d < 380


def test_build_features_columns_present(tiny_txns):
    feats = build_features(tiny_txns)
    for col in FEATURE_COLS:
        assert col in feats.columns
    assert len(feats) == len(tiny_txns)


def test_build_features_flags_the_takeover_row(tiny_txns):
    feats = build_features(tiny_txns).set_index("transaction_id")
    row = feats.loc[5]
    assert row["is_new_device"] == 1
    assert row["is_foreign_city"] == 1
    assert row["high_risk_category"] == 1
    assert row["amount_ratio"] > 3
    assert row["is_night"] == 1
    assert row["dist_from_home_km"] > 1000


def test_velocity_counts_increase_within_window():
    import pandas as pd
    base = dict(customer_id=9, amount=10.0, merchant_category="streaming",
                device_id="d", city="Sao Paulo", lat=-23.55, lon=-46.63)
    rows = [{**base, "transaction_id": i,
             "timestamp": f"2024-01-01 10:0{i}:00"} for i in range(5)]
    feats = build_features(pd.DataFrame(rows)).set_index("transaction_id")
    assert feats.loc[4, "txn_count_5min"] >= 4
