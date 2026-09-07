"""
Feedback loop (item 4 dos proximos passos).

O analista revisa um laudo e responde: era fraude ou foi falso positivo?
Essa decisao e gravada em data/feedback.jsonl e realimenta o treino:
`main.py retrain` reescreve os rotulos `is_fraud` das transacoes com
feedback e retreina o modelo.

Formato de cada linha (JSON):
  {"ts": "...", "dataset": "default", "transaction_id": 100123,
   "label": 1, "analyst": "ana"}
  label: 1 = fraude confirmada, 0 = falso positivo
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from . import config as cfg

FEEDBACK_PATH = cfg.DATA_DIR / "feedback.jsonl"


def record(transaction_id: int, label: int, dataset: str = "default",
           analyst: str | None = None) -> dict:
    if label not in (0, 1):
        raise ValueError("label deve ser 0 (falso positivo) ou 1 (fraude)")
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset,
        "transaction_id": int(transaction_id),
        "label": int(label),
        "analyst": analyst,
    }
    with FEEDBACK_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def load(dataset: str | None = None) -> pd.DataFrame:
    if not FEEDBACK_PATH.exists():
        return pd.DataFrame(columns=["ts", "dataset", "transaction_id", "label", "analyst"])
    rows = [json.loads(l) for l in FEEDBACK_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if dataset is not None and not df.empty:
        df = df[df["dataset"] == dataset]
    # se houver feedback repetido para a mesma transacao, o mais recente vence
    if not df.empty:
        df = df.sort_values("ts").drop_duplicates("transaction_id", keep="last")
    return df


def apply_to(feats: pd.DataFrame, dataset: str = "default") -> tuple[pd.DataFrame, int]:
    """Devolve uma copia de `feats` com is_fraud sobrescrito pelo feedback."""
    fb = load(dataset)
    if fb.empty:
        return feats, 0
    out = feats.copy()
    mapping = dict(zip(fb["transaction_id"], fb["label"]))
    mask = out["transaction_id"].isin(mapping)
    out.loc[mask, "is_fraud"] = out.loc[mask, "transaction_id"].map(mapping)
    return out, int(mask.sum())
