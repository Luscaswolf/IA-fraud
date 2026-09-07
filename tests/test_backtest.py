"""Backtest temporal: split respeita o tempo e produz metricas."""
import pandas as pd

from src import backtest
from src.features import build_features


def _labeled_stream(n=600):
    rows = []
    for i in range(n):
        fraud = int(i % 25 == 0)
        rows.append(dict(
            transaction_id=i,
            timestamp=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i),
            customer_id=i % 40,
            amount=800.0 if fraud else 90.0,
            merchant_category="cripto_exchange" if fraud else "supermercado",
            device_id="dev_new" if fraud else f"dev_{i % 40}",
            city="Miami" if fraud else "Sao Paulo",
            lat=25.76 if fraud else -23.55,
            lon=-80.19 if fraud else -46.63,
            is_fraud=fraud,
            fraud_pattern="account_takeover" if fraud else "none",
        ))
    return pd.DataFrame(rows)


def test_temporal_split_is_ordered():
    feats = build_features(_labeled_stream())
    train, test, split_ts = backtest.temporal_split(feats, 0.75)
    assert train["timestamp"].max() <= test["timestamp"].min()
    assert len(train) + len(test) == len(feats)


def test_run_reports_metrics(tmp_path):
    csv = tmp_path / "stream.csv"
    _labeled_stream().to_csv(csv, index=False)
    res = backtest.run(str(csv), train_frac=0.7)
    assert res["test_frauds"] > 0
    assert res["roc_auc"] is not None
    assert 0.0 <= res["roc_auc"] <= 1.0
    assert "roc_auc_random_split" in res
