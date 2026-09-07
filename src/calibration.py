"""
Calibracao dos limiares de decisao por CUSTO (item 5 dos proximos passos).

Hoje o sistema usa limiares fixos (0.4 = revisar, 0.7 = bloquear). Isso ignora
que o custo de deixar passar uma fraude != custo de bloquear um cliente legitimo.

Aqui varremos uma grade de limiares (t_review, t_block) sobre um conjunto
ROTULADO ja pontuado (`risk`, `is_fraud`, `amount`) e escolhemos o par que
minimiza o custo esperado por transacao, dado:

  - fraud_loss:      perda ao aprovar uma fraude          -> valor da transacao
  - friction_cost:   custo de bloquear um cliente legitimo -> venda perdida + atrito
  - review_cost:     custo de mandar um caso para analise manual

Politica avaliada:
  risk >= t_block            -> BLOQUEIA   (evita fraude; paga friction se era legitima)
  t_review <= risk < t_block -> REVISA     (paga review_cost; assume que a analise acerta)
  risk < t_review            -> APROVA     (perde o valor se era fraude)

O resultado e salvo em data/thresholds.json e consumido por
`narrative._recommendation`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as cfg

THRESHOLDS_PATH = cfg.DATA_DIR / "thresholds.json"

DEFAULT_REVIEW = 0.40
DEFAULT_BLOCK = 0.70


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        try:
            d = json.loads(THRESHOLDS_PATH.read_text())
            return {"review": float(d["review"]), "block": float(d["block"])}
        except Exception:
            pass
    return {"review": DEFAULT_REVIEW, "block": DEFAULT_BLOCK}


def _policy_cost(risk, is_fraud, amount, t_review, t_block,
                 friction_cost, review_cost) -> float:
    block = risk >= t_block
    review = (risk >= t_review) & (risk < t_block)
    approve = risk < t_review

    cost = 0.0
    # aprovou fraude -> perde o valor
    cost += float(amount[approve & (is_fraud == 1)].sum())
    # bloqueou legitima -> atrito
    cost += float(friction_cost * ((block & (is_fraud == 0)).sum()))
    # mandou para revisao -> custo operacional (fraude ou nao)
    cost += float(review_cost * review.sum())
    return cost


def calibrate(scored: pd.DataFrame,
              friction_cost: float | None = None,
              review_cost: float = 8.0,
              grid: int = 41) -> dict:
    """Retorna o melhor par de limiares + o custo esperado por transacao."""
    if "is_fraud" not in scored.columns:
        raise ValueError("calibracao exige um dataset rotulado (coluna is_fraud).")

    risk = scored["risk"].to_numpy()
    is_fraud = scored["is_fraud"].to_numpy()
    amount = scored["amount"].to_numpy()
    n = len(scored)

    # default: atrito = 1x o ticket medio das transacoes legitimas
    if friction_cost is None:
        friction_cost = float(scored.loc[scored.is_fraud == 0, "amount"].mean())

    baseline_approve_all = float(amount[is_fraud == 1].sum()) / n
    baseline_block_all = float(friction_cost * (is_fraud == 0).sum()) / n

    space = np.linspace(0.0, 1.0, grid)
    best = None
    for tr in space:
        for tb in space:
            if tb < tr:
                continue
            c = _policy_cost(risk, is_fraud, amount, tr, tb,
                             friction_cost, review_cost) / n
            if best is None or c < best["expected_cost_per_txn"]:
                best = {"review": round(float(tr), 3),
                        "block": round(float(tb), 3),
                        "expected_cost_per_txn": round(c, 4)}

    assert best is not None
    best.update({
        "friction_cost": round(friction_cost, 2),
        "review_cost": review_cost,
        "baseline_cost_approve_all": round(baseline_approve_all, 4),
        "baseline_cost_block_all": round(baseline_block_all, 4),
        "n_transactions": n,
    })
    return best


def save_thresholds(result: dict) -> None:
    THRESHOLDS_PATH.write_text(json.dumps(result, indent=2))
