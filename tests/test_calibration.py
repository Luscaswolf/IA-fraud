"""Calibracao de limiares por custo."""
import numpy as np
import pandas as pd

from src import calibration


def _synthetic_scored(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    is_fraud = (rng.random(n) < 0.02).astype(int)
    # fraudes tendem a ter risco alto; legitimas, baixo
    risk = np.clip(np.where(is_fraud == 1,
                            rng.normal(0.85, 0.1, n),
                            rng.normal(0.15, 0.12, n)), 0, 1)
    amount = np.where(is_fraud == 1, rng.uniform(500, 3000, n), rng.uniform(20, 300, n))
    return pd.DataFrame({"risk": risk, "is_fraud": is_fraud, "amount": amount})


def test_thresholds_ordered_and_bounded():
    res = calibration.calibrate(_synthetic_scored(), grid=21)
    assert 0.0 <= res["review"] <= res["block"] <= 1.0
    assert res["expected_cost_per_txn"] >= 0


def test_calibrated_beats_approve_all():
    res = calibration.calibrate(_synthetic_scored(), grid=21)
    assert res["expected_cost_per_txn"] <= res["baseline_cost_approve_all"]


def test_load_thresholds_default_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "THRESHOLDS_PATH", tmp_path / "nope.json")
    th = calibration.load_thresholds()
    assert th == {"review": calibration.DEFAULT_REVIEW, "block": calibration.DEFAULT_BLOCK}


def test_save_and_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "THRESHOLDS_PATH", tmp_path / "th.json")
    res = calibration.calibrate(_synthetic_scored(), grid=11)
    calibration.save_thresholds(res)
    th = calibration.load_thresholds()
    assert th["block"] == res["block"]
