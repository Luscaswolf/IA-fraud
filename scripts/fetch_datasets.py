"""
Baixa datasets REAIS de fraude do Hugging Face e adapta ao esquema do projeto.

Dataset: pointe77/credit-card-transaction  (base "Sparkov")
  Transacoes reais de cartao de credito, com clientes recorrentes,
  categoria, valor, localizacao e rotulo is_fraud. Ideal para o modelo
  de investigacao baseado em HISTORICO do cliente.

Saida: data/real_creditcard.csv (+ _scored.csv) -> aparece na interface.
"""
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config as cfg          # noqa: E402
from src import model as model_mod     # noqa: E402
from src.features import build_features  # noqa: E402

URL = ("https://huggingface.co/datasets/pointe77/credit-card-transaction/"
       "resolve/main/credit_card_transaction_train.csv")

N_ROWS = 200_000   # janela cronologica (preserva o historico por cliente)


def fetch() -> pd.DataFrame:
    print("Baixando amostra do dataset real de cartao de credito...")
    cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt",
            "city", "merch_lat", "merch_long", "is_fraud"]
    df = pd.read_csv(URL, usecols=cols, nrows=N_ROWS)
    print(f"  linhas lidas: {len(df):,}  fraudes: {int(df.is_fraud.sum()):,}")
    return df


def adapt(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["transaction_id"] = range(500000, 500000 + len(df))
    out["timestamp"] = pd.to_datetime(df["trans_date_trans_time"])
    out["customer_id"] = df["cc_num"].astype("int64")
    out["amount"] = df["amt"].round(2)
    out["merchant_category"] = df["category"].str.replace("_", " ")
    # a base nao possui dispositivo: um cartao = um dispositivo estavel
    out["device_id"] = "card_" + df["cc_num"].astype(str).str[-6:]
    out["city"] = df["city"]
    out["lat"] = df["merch_lat"]        # local da transacao
    out["lon"] = df["merch_long"]
    out["is_fraud"] = df["is_fraud"].astype(int)
    out["fraud_pattern"] = out["is_fraud"].map({1: "fraude_confirmada", 0: "none"})
    return out.sort_values("timestamp").reset_index(drop=True)


def main():
    t = time.time()
    df = adapt(fetch())
    raw_path = cfg.DATA_DIR / "real_creditcard.csv"
    df.to_csv(raw_path, index=False)

    print("Calculando features e score de risco...")
    feats = build_features(df)
    clf = joblib.load(cfg.MODEL_PATH)
    feats["risk"] = model_mod.score(clf, feats)
    scored = cfg.DATA_DIR / "real_creditcard_scored.csv"
    feats.to_csv(scored, index=False)

    n_fraud = int(df.is_fraud.sum())
    print(f"\nConcluido em {time.time()-t:.0f}s")
    print(f"  transacoes:  {len(df):,}")
    print(f"  clientes:    {df.customer_id.nunique():,}")
    print(f"  fraudes:     {n_fraud:,} ({n_fraud/len(df):.2%})")
    print(f"  alto risco:  {int((feats.risk>=0.7).sum()):,}")
    print(f"  risco medio -> fraude: {feats[feats.is_fraud==1].risk.mean():.2f} "
          f"| legitima: {feats[feats.is_fraud==0].risk.mean():.2f}")
    print(f"  salvo: {scored}")


if __name__ == "__main__":
    main()
