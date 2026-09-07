"""Configuracao central da POC."""
from pathlib import Path

# Diretorios
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# Arquivos gerados
CUSTOMERS_CSV = DATA_DIR / "customers.csv"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
MODEL_PATH = DATA_DIR / "fraud_model.joblib"

# Parametros de geracao
SEED = 42
N_CUSTOMERS = 2_000
N_TRANSACTIONS = 100_000
FRAUD_RATE = 0.012  # ~1.2% de fraudes injetadas

# Catalogos
CITIES = [
    ("Sao Paulo", -23.55, -46.63),
    ("Rio de Janeiro", -22.90, -43.20),
    ("Belo Horizonte", -19.92, -43.94),
    ("Curitiba", -25.42, -49.27),
    ("Porto Alegre", -30.03, -51.23),
    ("Recife", -8.05, -34.90),
    ("Salvador", -12.97, -38.50),
    ("Fortaleza", -3.73, -38.52),
    ("Brasilia", -15.79, -47.88),
    ("Manaus", -3.10, -60.02),
]

FOREIGN_CITIES = [
    ("Miami", 25.76, -80.19),
    ("Lisboa", 38.72, -9.13),
    ("Buenos Aires", -34.60, -58.38),
    ("Cidade do Mexico", 19.43, -99.13),
    ("Lagos", 6.52, 3.37),
]

MERCHANT_CATEGORIES = [
    "supermercado", "restaurante", "posto_combustivel", "farmacia",
    "eletronicos", "vestuario", "streaming", "transporte", "viagem",
    "joalheria", "cripto_exchange", "aposta_online", "transferencia_p2p",
]

# Categorias tipicamente exploradas em fraude (maior risco base)
HIGH_RISK_CATEGORIES = {"eletronicos", "joalheria", "cripto_exchange", "aposta_online", "transferencia_p2p"}

# Tipos de padrao de fraude injetados (motivo conhecido = ground truth)
FRAUD_PATTERNS = [
    "high_amount",        # valor muito acima da media do cliente
    "new_device",         # dispositivo nunca visto
    "impossible_travel",  # localizacao geografica impossivel no tempo
    "velocity_burst",     # varias transacoes em poucos minutos
    "card_testing",       # varias transacoes pequenas em sequencia
    "account_takeover",   # combinacao: novo device + nova location + valor alto
    "risky_merchant",     # categoria de alto risco em horario atipico
]
