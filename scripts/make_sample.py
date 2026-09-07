"""
Gera um arquivo de teste REAL e coerente para importar na interface.

Produz um CSV pequeno (dezenas de clientes, algumas centenas de transacoes)
com historico consistente por cliente e padroes de fraude reais injetados e
rotulados (colunas is_fraud / fraud_pattern), para o usuario validar o motivo.

Saida: sample_transactions.csv (na raiz do projeto)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generator import generate  # noqa: E402

# conjunto pequeno, porem com historico suficiente para velocidade/deslocamento
_, txns = generate(n_customers=30, n_transactions=400, fraud_rate=0.14)

cols = ["transaction_id", "timestamp", "customer_id", "amount",
        "merchant_category", "device_id", "city", "lat", "lon",
        "is_fraud", "fraud_pattern"]

out = ROOT / "sample_transactions.csv"
txns[cols].to_csv(out, index=False)

n_fraud = int(txns["is_fraud"].sum())
print(f"Arquivo de teste gerado: {out}")
print(f"  transacoes: {len(txns)}")
print(f"  fraudes:    {n_fraud}")
print("  padroes injetados:")
print(txns[txns.is_fraud == 1]["fraud_pattern"].value_counts().to_string())
