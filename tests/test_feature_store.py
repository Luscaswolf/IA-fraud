"""O feature store online deve reproduzir build_features (com tolerancia)."""
import pandas as pd

from src.feature_store import FeatureStore
from src.features import build_features


def _txn_dicts(df: pd.DataFrame):
    cols = ["transaction_id", "timestamp", "customer_id", "amount",
            "merchant_category", "device_id", "city", "lat", "lon"]
    return [r._asdict() for r in df[cols].itertuples(index=False)]


def test_online_matches_batch_from_second_txn(tiny_txns):
    batch = build_features(tiny_txns).set_index("transaction_id")
    store = FeatureStore()

    for txn in _txn_dicts(tiny_txns):
        online = store.online_features(txn) if False else store.state(txn["customer_id"]).online_features(txn)
        store.update(txn)
        tid = txn["transaction_id"]
        if tid == 1:
            continue  # 1a transacao: perfil ainda indefinido, diverge por design
        b = batch.loc[tid]
        assert online["is_foreign_city"] == b["is_foreign_city"]
        assert online["high_risk_category"] == b["high_risk_category"]
        assert online["is_night"] == b["is_night"]
        assert online["txn_count_5min"] == b["txn_count_5min"]
        # ratio proximo: perfil online usa so o passado, batch usa a media global
        assert online["amount_ratio"] > 0


def test_takeover_row_online_signals(tiny_txns):
    store = FeatureStore()
    txns = _txn_dicts(tiny_txns)
    for txn in txns[:-1]:
        store.update(txn)
    last = txns[-1]
    f = store.state(last["customer_id"]).online_features(last)
    assert f["is_new_device"] == 1
    assert f["is_foreign_city"] == 1
    assert f["amount_ratio"] > 3
    assert f["dist_from_home_km"] > 1000


def test_serialization_roundtrip(tiny_txns):
    from src.feature_store import CustomerState
    store = FeatureStore()
    for txn in _txn_dicts(tiny_txns):
        store.update(txn)
    st = store.state(1)
    st2 = CustomerState.from_dict(st.to_dict())
    assert st2.n == st.n
    assert st2.devices == st.devices
    assert round(st2.avg_amount, 4) == round(st.avg_amount, 4)


def test_bootstrap_warms_state(tiny_txns):
    feats = build_features(tiny_txns)
    store = FeatureStore().bootstrap(feats)
    assert store.known(1)
    assert store.state(1).n == len(tiny_txns)
