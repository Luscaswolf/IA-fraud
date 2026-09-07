"""
Modelo de deteccao de fraude (score de risco).

Papel na POC: e o 'primeiro filtro'. Recebe 100k transacoes e devolve
um risco 0-100%. Transacoes de alto risco sao encaminhadas ao Agente
Investigador para analise aprofundada.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from . import config as cfg
from .features import FEATURE_COLS, build_features


def train(feats: pd.DataFrame):
    X = feats[FEATURE_COLS].fillna(0)
    y = feats["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=cfg.SEED, stratify=y
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced",
        n_jobs=-1, random_state=cfg.SEED,
    )
    clf.fit(X_tr, y_tr)

    proba = clf.predict_proba(X_te)[:, 1]
    print(f"ROC-AUC:  {roc_auc_score(y_te, proba):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_te, proba):.4f}")
    print(classification_report(y_te, (proba >= 0.5).astype(int), digits=3))

    print("Importancia das features:")
    imp = sorted(zip(FEATURE_COLS, clf.feature_importances_),
                 key=lambda x: -x[1])
    for name, val in imp:
        print(f"  {name:20s} {val:.3f}")
    return clf


def load():
    return joblib.load(cfg.MODEL_PATH)


def score(clf, feats: pd.DataFrame) -> pd.Series:
    proba = clf.predict_proba(feats[FEATURE_COLS].fillna(0))[:, 1]
    return pd.Series(proba, index=feats.index, name="risk")


def main():
    print("Carregando dados e construindo features...")
    txns = pd.read_csv(cfg.TRANSACTIONS_CSV)
    customers = pd.read_csv(cfg.CUSTOMERS_CSV)
    feats = build_features(txns, customers)

    print("Treinando modelo...")
    clf = train(feats)
    joblib.dump(clf, cfg.MODEL_PATH)
    print(f"Modelo salvo em: {cfg.MODEL_PATH}")


if __name__ == "__main__":
    main()
