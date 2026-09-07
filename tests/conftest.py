"""Fixtures compartilhadas dos testes."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def tiny_txns() -> pd.DataFrame:
    """Historico pequeno de 1 cliente: 4 compras normais + 1 obvia de takeover."""
    base = dict(customer_id=1, merchant_category="supermercado",
                device_id="dev_home", city="Sao Paulo", lat=-23.55, lon=-46.63)
    rows = [
        {**base, "transaction_id": 1, "timestamp": "2024-01-01 10:00:00", "amount": 100.0},
        {**base, "transaction_id": 2, "timestamp": "2024-01-02 11:00:00", "amount": 120.0},
        {**base, "transaction_id": 3, "timestamp": "2024-01-03 12:00:00", "amount": 90.0},
        {**base, "transaction_id": 4, "timestamp": "2024-01-04 13:00:00", "amount": 110.0},
        # account takeover: device novo + cidade estrangeira + valor 8x
        {**base, "transaction_id": 5, "timestamp": "2024-01-05 03:00:00", "amount": 850.0,
         "device_id": "dev_new", "city": "Miami", "lat": 25.76, "lon": -80.19,
         "merchant_category": "cripto_exchange"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def model_available() -> bool:
    from src import config as cfg
    return cfg.MODEL_PATH.exists()
