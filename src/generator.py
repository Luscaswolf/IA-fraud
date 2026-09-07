"""
Gerador de dados sinteticos com padroes de fraude INJETADOS e ROTULADOS.

Cada transacao fraudulenta carrega:
  - is_fraud = 1
  - fraud_pattern = motivo real (ground truth)

Isso permite avaliar depois se o Agente Investigador encontrou a causa correta.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

from . import config as cfg


def _haversine(lat1, lon1, lat2, lon2):
    """Distancia em km entre dois pontos."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def generate_customers(n: int, faker: Faker) -> pd.DataFrame:
    rows = []
    for cid in range(1, n + 1):
        home = random.choice(cfg.CITIES)
        # perfil de gasto do cliente
        avg = round(random.uniform(40, 600), 2)
        rows.append({
            "customer_id": cid,
            "name": faker.name(),
            "home_city": home[0],
            "home_lat": home[1],
            "home_lon": home[2],
            "avg_amount": avg,
            "std_amount": round(avg * random.uniform(0.2, 0.5), 2),
            "primary_device": f"dev_{faker.uuid4()[:8]}",
            "account_age_days": random.randint(30, 3000),
        })
    return pd.DataFrame(rows)


def _base_transaction(cust, ts, faker):
    """Transacao legitima 'normal' para o cliente."""
    amount = max(1.0, round(np.random.normal(cust["avg_amount"], cust["std_amount"]), 2))
    return {
        "timestamp": ts,
        "customer_id": cust["customer_id"],
        "amount": amount,
        "merchant_category": random.choice(cfg.MERCHANT_CATEGORIES),
        "merchant_id": f"m_{random.randint(1000, 9999)}",
        "device_id": cust["primary_device"] if random.random() < 0.92
        else f"dev_{faker.uuid4()[:8]}",
        "city": cust["home_city"],
        "lat": cust["home_lat"] + random.uniform(-0.05, 0.05),
        "lon": cust["home_lon"] + random.uniform(-0.05, 0.05),
        "is_fraud": 0,
        "fraud_pattern": "none",
    }


def _inject(pattern, cust, ts, faker):
    """Gera 1..N transacoes fraudulentas de um padrao especifico."""
    txns = []
    if pattern == "high_amount":
        t = _base_transaction(cust, ts, faker)
        t["amount"] = round(cust["avg_amount"] * random.uniform(5.0, 12.0), 2)
        t["merchant_category"] = random.choice(list(cfg.HIGH_RISK_CATEGORIES))
        t.update(is_fraud=1, fraud_pattern="high_amount")
        txns.append(t)

    elif pattern == "new_device":
        t = _base_transaction(cust, ts, faker)
        t["device_id"] = f"dev_{faker.uuid4()[:8]}"
        t["amount"] = round(cust["avg_amount"] * random.uniform(1.5, 3.0), 2)
        t.update(is_fraud=1, fraud_pattern="new_device")
        txns.append(t)

    elif pattern == "impossible_travel":
        city = random.choice(cfg.FOREIGN_CITIES)
        t = _base_transaction(cust, ts, faker)
        t["city"], t["lat"], t["lon"] = city[0], city[1], city[2]
        t["device_id"] = f"dev_{faker.uuid4()[:8]}"
        t.update(is_fraud=1, fraud_pattern="impossible_travel")
        txns.append(t)

    elif pattern == "velocity_burst":
        n = random.randint(4, 7)
        for i in range(n):
            t = _base_transaction(cust, ts + timedelta(seconds=i * random.randint(15, 45)), faker)
            t["amount"] = round(cust["avg_amount"] * random.uniform(1.2, 2.5), 2)
            t.update(is_fraud=1, fraud_pattern="velocity_burst")
            txns.append(t)

    elif pattern == "card_testing":
        n = random.randint(6, 12)
        for i in range(n):
            t = _base_transaction(cust, ts + timedelta(seconds=i * random.randint(5, 20)), faker)
            t["amount"] = round(random.uniform(0.5, 4.0), 2)  # micro valores
            t["merchant_category"] = "streaming"
            t.update(is_fraud=1, fraud_pattern="card_testing")
            txns.append(t)

    elif pattern == "account_takeover":
        city = random.choice(cfg.FOREIGN_CITIES)
        t = _base_transaction(cust, ts, faker)
        t["device_id"] = f"dev_{faker.uuid4()[:8]}"
        t["city"], t["lat"], t["lon"] = city[0], city[1], city[2]
        t["amount"] = round(cust["avg_amount"] * random.uniform(4.0, 9.0), 2)
        t["merchant_category"] = random.choice(list(cfg.HIGH_RISK_CATEGORIES))
        t.update(is_fraud=1, fraud_pattern="account_takeover")
        txns.append(t)

    elif pattern == "risky_merchant":
        # horario atipico (madrugada)
        night = ts.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))
        t = _base_transaction(cust, night, faker)
        t["merchant_category"] = random.choice(list(cfg.HIGH_RISK_CATEGORIES))
        t["amount"] = round(cust["avg_amount"] * random.uniform(2.0, 4.0), 2)
        t.update(is_fraud=1, fraud_pattern="risky_merchant")
        txns.append(t)

    return txns


def generate(n_customers=cfg.N_CUSTOMERS, n_transactions=cfg.N_TRANSACTIONS,
             fraud_rate=cfg.FRAUD_RATE):
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    faker = Faker("pt_BR")
    Faker.seed(cfg.SEED)

    customers = generate_customers(n_customers, faker)
    cust_records = customers.to_dict("index")
    cust_list = list(cust_records.values())

    start = datetime(2024, 1, 1)
    horizon_days = 90

    txns = []
    n_fraud_target = int(n_transactions * fraud_rate)
    n_legit = n_transactions - n_fraud_target

    # transacoes legitimas
    for _ in range(n_legit):
        cust = random.choice(cust_list)
        ts = start + timedelta(
            days=random.randint(0, horizon_days),
            hours=random.randint(6, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        txns.append(_base_transaction(cust, ts, faker))

    # transacoes fraudulentas (por padrao)
    injected = 0
    while injected < n_fraud_target:
        pattern = random.choice(cfg.FRAUD_PATTERNS)
        cust = random.choice(cust_list)
        ts = start + timedelta(
            days=random.randint(0, horizon_days),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )
        batch = _inject(pattern, cust, ts, faker)
        txns.extend(batch)
        injected += len(batch)

    df = pd.DataFrame(txns).sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "transaction_id", range(100000, 100000 + len(df)))
    return customers, df


def main():
    print("Gerando dados sinteticos...")
    customers, txns = generate()
    customers.to_csv(cfg.CUSTOMERS_CSV, index=False)
    txns.to_csv(cfg.TRANSACTIONS_CSV, index=False)
    n_fraud = int(txns["is_fraud"].sum())
    print(f"  clientes:     {len(customers):,}")
    print(f"  transacoes:   {len(txns):,}")
    print(f"  fraudes:      {n_fraud:,} ({n_fraud / len(txns):.2%})")
    print("  distribuicao de padroes de fraude:")
    print(txns[txns.is_fraud == 1]["fraud_pattern"].value_counts().to_string())
    print(f"  salvo em: {cfg.TRANSACTIONS_CSV}")


if __name__ == "__main__":
    main()
