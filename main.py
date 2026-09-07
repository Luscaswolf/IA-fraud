"""
AI Fraud Investigator - CLI da POC.

Uso:
  python main.py generate          # gera dados sinteticos rotulados
  python main.py train             # treina modelo de deteccao + salva scores
  python main.py investigate <id>  # investiga uma transacao especifica
  python main.py narrate <id>      # laudo em linguagem natural (LLM/template)
  python main.py demo [N]          # investiga as N transacoes de maior risco
  python main.py evaluate [N]      # mede se o agente acerta o MOTIVO da fraude
"""
from __future__ import annotations

import sys

import pandas as pd

from src import config as cfg
from src import generator, model, report
from src.features import build_features
from src.investigator import FraudInvestigator

SCORED_CSV = cfg.DATA_DIR / "transactions_scored.csv"


def _load_scored() -> pd.DataFrame:
    if not SCORED_CSV.exists():
        raise SystemExit("Execute 'python main.py train' primeiro.")
    return pd.read_csv(SCORED_CSV)


def cmd_generate():
    generator.main()


def cmd_train():
    txns = pd.read_csv(cfg.TRANSACTIONS_CSV)
    customers = pd.read_csv(cfg.CUSTOMERS_CSV)
    print("Construindo features...")
    feats = build_features(txns, customers)
    print("Treinando modelo...")
    clf = model.train(feats)
    import joblib
    joblib.dump(clf, cfg.MODEL_PATH)
    feats["risk"] = model.score(clf, feats)
    feats.to_csv(SCORED_CSV, index=False)
    print(f"\nScores salvos em: {SCORED_CSV}")


def cmd_investigate(tid: int):
    feats = _load_scored()
    inv_engine = FraudInvestigator(feats)
    result = inv_engine.investigate(tid)
    row = feats[feats.transaction_id == tid].iloc[0]
    true = row["fraud_pattern"] if row["is_fraud"] == 1 else None
    report.render_rich(result, true)


def cmd_narrate(tid: int):
    from src.narrative import narrate
    feats = _load_scored()
    inv_engine = FraudInvestigator(feats)
    result = inv_engine.investigate(tid)
    row = feats[feats.transaction_id == tid].iloc[0]
    true = row["fraud_pattern"] if row["is_fraud"] == 1 else None
    payload = narrate(result, true)
    print(f"\n[engine de narrativa: {payload['narrative_engine']}]\n")
    print(payload["narrative"])
    if true:
        ok = "CORRETO" if payload["correct_motive"] else "DIVERGENTE"
        print(f"\n(motivo real: {true} -> {ok})")


def cmd_demo(n=5):
    feats = _load_scored()
    inv_engine = FraudInvestigator(feats)
    top = feats.sort_values("risk", ascending=False).head(n)
    for _, row in top.iterrows():
        result = inv_engine.investigate(int(row["transaction_id"]))
        true = row["fraud_pattern"] if row["is_fraud"] == 1 else None
        report.render_rich(result, true)
        print()


def cmd_evaluate(n=300):
    feats = _load_scored()
    inv_engine = FraudInvestigator(feats)
    # avalia sobre as transacoes de maior risco que sao realmente fraude
    top = feats.sort_values("risk", ascending=False).head(n)
    frauds = top[top.is_fraud == 1]
    hits, total = 0, 0
    per_pattern = {}
    for _, row in frauds.iterrows():
        result = inv_engine.investigate(int(row["transaction_id"]))
        true = row["fraud_pattern"]
        ok = result.inferred_pattern == true
        hits += int(ok)
        total += 1
        d = per_pattern.setdefault(true, [0, 0])
        d[0] += int(ok)
        d[1] += 1
    print(f"\nAvaliacao do MOTIVO inferido pelo agente (top {n} por risco)")
    print(f"Fraudes analisadas: {total}")
    if total:
        print(f"Acerto do motivo:   {hits}/{total} = {hits/total:.1%}\n")
    print(f"{'padrao':22s} {'acerto':>10s}")
    for pat, (h, t) in sorted(per_pattern.items()):
        print(f"{pat:22s} {h}/{t} = {h/t:.0%}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cmd = sys.argv[1]
    if cmd == "generate":
        cmd_generate()
    elif cmd == "train":
        cmd_train()
    elif cmd == "investigate":
        cmd_investigate(int(sys.argv[2]))
    elif cmd == "narrate":
        cmd_narrate(int(sys.argv[2]))
    elif cmd == "demo":
        cmd_demo(int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif cmd == "evaluate":
        cmd_evaluate(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
