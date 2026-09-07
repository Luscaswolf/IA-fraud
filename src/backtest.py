"""
Backtesting temporal (itens 3 e 6 dos proximos passos).

Problema do `model.train` atual: usa `train_test_split` aleatorio, entao
transacoes do MESMO cliente (e ate da mesma rajada de fraude) caem nos dois
lados -> vazamento -> as metricas 0.999 sao otimistas.

Aqui: ordena por tempo, treina SO com o passado, testa SO no futuro.
Funciona tanto no dataset sintetico quanto num CSV real (ex.: Sparkov):

    python main.py backtest                         # sintetico
    python main.py backtest data/real_creditcard.csv  # dados reais
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as cfg
from .features import FEATURE_COLS, build_features


def temporal_split(feats: pd.DataFrame, train_frac: float = 0.75):
    """Divide por tempo: primeiros `train_frac` no treino, resto no teste."""
    df = feats.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    split_ts = df.loc[cut, "timestamp"]
    train = df.iloc[:cut]
    test = df.iloc[cut:]
    return train, test, split_ts


def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if k <= 0 or len(scores) == 0:
        return float("nan")
    idx = np.argsort(scores)[::-1][:k]
    return float(y_true[idx].mean())


def run(csv_path: str | Path | None = None, train_frac: float = 0.75) -> dict:
    if csv_path is None:
        txns = pd.read_csv(cfg.TRANSACTIONS_CSV)
        customers = pd.read_csv(cfg.CUSTOMERS_CSV)
        feats = build_features(txns, customers)
        source = "sintetico"
    else:
        df = pd.read_csv(csv_path)
        feats = build_features(df) if "amount_ratio" not in df.columns else df
        source = str(csv_path)

    if "is_fraud" not in feats.columns:
        raise SystemExit("O CSV precisa da coluna is_fraud para o backtest.")

    train, test, split_ts = temporal_split(feats, train_frac)
    Xtr, ytr = train[FEATURE_COLS].fillna(0), train["is_fraud"].astype(int)
    Xte, yte = test[FEATURE_COLS].fillna(0), test["is_fraud"].astype(int)

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=cfg.SEED,
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    yte_arr = yte.to_numpy()

    n_pos = int(yte_arr.sum())
    res = {
        "source": source,
        "split_timestamp": str(split_ts),
        "train_rows": len(train), "test_rows": len(test),
        "test_frauds": n_pos,
        "roc_auc": round(roc_auc_score(yte_arr, proba), 4) if n_pos else None,
        "pr_auc": round(average_precision_score(yte_arr, proba), 4) if n_pos else None,
        "precision_at_100": round(_precision_at_k(yte_arr, proba, 100), 4),
        "precision_at_500": round(_precision_at_k(yte_arr, proba, 500), 4),
        "recall_at_1pct": None,
    }
    k1 = max(1, int(0.01 * len(test)))
    if n_pos:
        top_idx = np.argsort(proba)[::-1][:k1]
        res["recall_at_1pct"] = round(float(yte_arr[top_idx].sum() / n_pos), 4)

    # comparacao honesta: split aleatorio no mesmo dado
    from sklearn.model_selection import train_test_split
    Xa, Xb, ya, yb = train_test_split(
        feats[FEATURE_COLS].fillna(0), feats["is_fraud"].astype(int),
        test_size=0.25, random_state=cfg.SEED, stratify=feats["is_fraud"],
    )
    clf2 = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=cfg.SEED,
    ).fit(Xa, ya)
    p2 = clf2.predict_proba(Xb)[:, 1]
    res["roc_auc_random_split"] = round(roc_auc_score(yb, p2), 4)
    res["pr_auc_random_split"] = round(average_precision_score(yb, p2), 4)
    return res


def print_report(res: dict) -> None:
    print("\n=== Backtest temporal ===")
    print(f"  fonte:              {res['source']}")
    print(f"  corte temporal:     {res['split_timestamp']}")
    print(f"  treino / teste:     {res['train_rows']:,} / {res['test_rows']:,} linhas")
    print(f"  fraudes no teste:   {res['test_frauds']:,}")
    print(f"  ROC-AUC (futuro):   {res['roc_auc']}")
    print(f"  PR-AUC  (futuro):   {res['pr_auc']}")
    print(f"  precisao@100:       {res['precision_at_100']}")
    print(f"  precisao@500:       {res['precision_at_500']}")
    print(f"  recall no top 1%:   {res['recall_at_1pct']}")
    print(f"\n  (referencia) split aleatorio -> ROC-AUC {res['roc_auc_random_split']} "
          f"/ PR-AUC {res['pr_auc_random_split']}")
    print("  Se o split aleatorio for muito melhor que o temporal, ha vazamento.\n")
