"""O agente deve inferir o motivo correto em casos obvios."""
import pandas as pd

from src.features import build_features
from src.investigator import FraudInvestigator


def test_infers_account_takeover(tiny_txns):
    feats = build_features(tiny_txns)
    feats["risk"] = 0.9
    eng = FraudInvestigator(feats)
    inv = eng.investigate(5)
    assert inv.inferred_pattern == "account_takeover"
    assert inv.fraud_probability > 0.6
    assert any(e.pattern == "account_takeover" for e in inv.evidences)


def test_infers_card_testing():
    base = dict(customer_id=7, merchant_category="streaming",
                device_id="d", city="Sao Paulo", lat=-23.55, lon=-46.63)
    hist = [{**base, "transaction_id": i, "amount": 100.0,
             "timestamp": f"2024-01-0{i+1} 10:00:00"} for i in range(1, 5)]
    burst = [{**base, "transaction_id": 100 + i, "amount": 2.0,
              "timestamp": f"2024-02-01 10:00:{i*8:02d}"} for i in range(8)]
    feats = build_features(pd.DataFrame(hist + burst))
    feats["risk"] = 0.8
    eng = FraudInvestigator(feats)
    inv = eng.investigate(107)
    assert inv.inferred_pattern == "card_testing"


def test_investigate_row_online_path(tiny_txns):
    feats = build_features(tiny_txns)
    feats["risk"] = 0.0
    eng = FraudInvestigator(feats)
    row = feats.set_index("transaction_id", drop=False).loc[5]
    inv = eng.investigate_row(row, past=feats[feats.transaction_id < 5])
    assert inv.inferred_pattern == "account_takeover"
