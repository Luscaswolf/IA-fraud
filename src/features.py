"""
Engenharia de features: transforma cada transacao em um vetor contextual
relativo ao historico do cliente (base tanto para o modelo quanto para o agente).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def _haversine_series(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


FEATURE_COLS = [
    "amount",
    "amount_ratio",       # valor / media do cliente
    "amount_zscore",      # desvios-padrao acima da media
    "hour",
    "is_night",
    "dist_from_home_km",
    "is_new_device",
    "is_foreign_city",
    "high_risk_category",
    "txn_count_5min",     # velocidade: transacoes nos ultimos 5 min
    "txn_count_1h",
    "sec_since_last",     # segundos desde a transacao anterior do cliente
    "km_per_hour",        # velocidade geografica implicita (impossible travel)
]


def derive_customer_profiles(txns: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva o 'perfil normal' de cada cliente a partir das proprias transacoes.
    Usado quando o usuario envia um CSV sem tabela de clientes separada.
      - avg_amount / std_amount : media e desvio dos valores do cliente
      - primary_device          : dispositivo mais frequente
      - home_lat / home_lon      : localizacao mediana (aproxima a residencia)
    """
    df = txns.copy()
    prof = df.groupby("customer_id").agg(
        avg_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        home_lat=("lat", "median"),
        home_lon=("lon", "median"),
    ).reset_index()
    prof["std_amount"] = prof["std_amount"].fillna(prof["avg_amount"] * 0.3).clip(lower=1)
    prof["avg_amount"] = prof["avg_amount"].clip(lower=1)
    dev = (df.groupby("customer_id")["device_id"]
             .agg(lambda s: s.value_counts().index[0]).reset_index()
             .rename(columns={"device_id": "primary_device"}))
    return prof.merge(dev, on="customer_id", how="left")


REQUIRED_COLS = ["transaction_id", "timestamp", "customer_id", "amount",
                 "merchant_category", "device_id", "city", "lat", "lon"]


def build_features(txns: pd.DataFrame, customers: pd.DataFrame | None = None) -> pd.DataFrame:
    df = txns.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # se nao houver tabela de clientes, deriva o perfil dos proprios dados
    if customers is None:
        customers = derive_customer_profiles(df)

    df = df.merge(
        customers[["customer_id", "home_lat", "home_lon", "avg_amount",
                   "std_amount", "primary_device"]],
        on="customer_id", how="left",
    )

    # --- features estaticas ---
    df["amount_ratio"] = df["amount"] / df["avg_amount"].clip(lower=1)
    df["amount_zscore"] = (df["amount"] - df["avg_amount"]) / df["std_amount"].clip(lower=1)
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["dist_from_home_km"] = _haversine_series(
        df["lat"], df["lon"], df["home_lat"], df["home_lon"]
    )
    df["is_new_device"] = (df["device_id"] != df["primary_device"]).astype(int)
    df["is_foreign_city"] = df["city"].isin([c[0] for c in cfg.FOREIGN_CITIES]).astype(int)
    df["high_risk_category"] = df["merchant_category"].isin(cfg.HIGH_RISK_CATEGORIES).astype(int)

    # --- features temporais por cliente (ordenadas) ---
    df = df.sort_values(["customer_id", "timestamp"]).reset_index(drop=True)

    parts = []
    for _, g in df.groupby("customer_id", sort=False):
        g = g.copy()
        g["sec_since_last"] = g["timestamp"].diff().dt.total_seconds().fillna(1e6)
        dist = _haversine_series(
            g["lat"], g["lon"], g["lat"].shift(), g["lon"].shift()
        ).fillna(0)
        hours = (g["sec_since_last"] / 3600).clip(lower=1e-4)
        g["km_per_hour"] = (dist / hours).fillna(0)
        idx = g.set_index("timestamp")
        g["txn_count_5min"] = idx["amount"].rolling("5min").count().values
        g["txn_count_1h"] = idx["amount"].rolling("1h").count().values
        parts.append(g)

    df = pd.concat(parts)
    df["km_per_hour"] = df["km_per_hour"].replace([np.inf, -np.inf], 0)

    return df.sort_values("transaction_id").reset_index(drop=True)


def main():
    txns = pd.read_csv(cfg.TRANSACTIONS_CSV)
    customers = pd.read_csv(cfg.CUSTOMERS_CSV)
    feats = build_features(txns, customers)
    print(feats[["transaction_id"] + FEATURE_COLS].describe().T.to_string())


if __name__ == "__main__":
    main()
