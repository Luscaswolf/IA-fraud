"""
Feature store online (janela deslizante por cliente).

Item 2 dos proximos passos: em vez de recalcular features varrendo o CSV
inteiro, mantemos por cliente um ESTADO INCREMENTAL:

  - estatisticas de valor (media/desvio via Welford)
  - dispositivos ja vistos
  - amostra de coordenadas (aproxima a residencia pela mediana)
  - janela das ultimas N transacoes (timestamps) para velocity
  - ultima transacao (para sec_since_last e km_per_hour / impossible travel)

`CustomerState.online_features()` devolve exatamente as colunas de
`features.FEATURE_COLS`, para alimentar o mesmo modelo e o mesmo agente.

Hoje o armazenamento e um dict em memoria (`FeatureStore`). A interface
(`snapshot` / `update`) foi desenhada para ser trocada por Redis/Cassandra
sem tocar no resto do codigo: cada `CustomerState` serializa para um dict
plano via `to_dict()` / `from_dict()`.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from . import config as cfg
from .features import FEATURE_COLS, haversine

WINDOW_SECONDS = 3600          # 1h de janela para contagem de velocity
MAX_COORDS = 200               # amostra de coordenadas para estimar a "casa"
NO_PREVIOUS = 1_000_000.0      # sentinela de sec_since_last quando nao ha anterior


@dataclass
class CustomerState:
    customer_id: int
    n: int = 0
    _mean: float = 0.0
    _m2: float = 0.0                       # soma dos quadrados (Welford)
    devices: set = field(default_factory=set)
    coords: deque = field(default_factory=lambda: deque(maxlen=MAX_COORDS))
    recent_ts: deque = field(default_factory=deque)  # timestamps (epoch s) da janela
    last_ts: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None

    # ---------- leitura ----------

    @property
    def avg_amount(self) -> float:
        return max(self._mean, 1.0) if self.n else 1.0

    @property
    def std_amount(self) -> float:
        if self.n >= 2:
            return max(math.sqrt(self._m2 / (self.n - 1)), 1.0)
        return max(self.avg_amount * 0.3, 1.0)

    @property
    def home_lat(self) -> float | None:
        if not self.coords:
            return None
        return float(pd.Series([c[0] for c in self.coords]).median())

    @property
    def home_lon(self) -> float | None:
        if not self.coords:
            return None
        return float(pd.Series([c[1] for c in self.coords]).median())

    def snapshot(self) -> dict:
        """Perfil 'normal' do cliente derivado do estado atual."""
        return {
            "customer_id": self.customer_id,
            "transactions_seen": self.n,
            "avg_amount": round(self.avg_amount, 2),
            "std_amount": round(self.std_amount, 2),
            "home_lat": self.home_lat,
            "home_lon": self.home_lon,
            "known_devices": len(self.devices),
        }

    # ---------- calculo das features online ----------

    def online_features(self, txn: dict) -> dict:
        """
        Recebe UMA transacao (dict com as colunas REQUIRED_COLS) e devolve o
        vetor FEATURE_COLS calculado contra o estado ATUAL (antes do update).
        """
        ts = pd.to_datetime(txn["timestamp"])
        epoch = ts.timestamp()
        amount = float(txn["amount"])
        lat, lon = float(txn["lat"]), float(txn["lon"])

        avg, std = self.avg_amount, self.std_amount
        home_lat = self.home_lat if self.home_lat is not None else lat
        home_lon = self.home_lon if self.home_lon is not None else lon

        if self.last_ts is None:
            sec_since_last = NO_PREVIOUS
            km_per_hour = 0.0
        else:
            sec_since_last = max(epoch - self.last_ts, 0.0)
            dist_prev = haversine(lat, lon, self.last_lat, self.last_lon)
            hours = max(sec_since_last / 3600.0, 1e-4)
            km_per_hour = dist_prev / hours

        # velocity: transacoes na janela + a atual
        cutoff_5m = epoch - 300
        cutoff_1h = epoch - WINDOW_SECONDS
        c5 = sum(1 for t in self.recent_ts if t >= cutoff_5m) + 1
        c1h = sum(1 for t in self.recent_ts if t >= cutoff_1h) + 1

        feats = {
            "amount": amount,
            "amount_ratio": amount / max(avg, 1.0),
            "amount_zscore": (amount - avg) / max(std, 1.0),
            "hour": ts.hour,
            "is_night": int(0 <= ts.hour <= 5),
            "dist_from_home_km": haversine(lat, lon, home_lat, home_lon),
            "is_new_device": int(txn["device_id"] not in self.devices),
            "is_foreign_city": int(txn["city"] in {c[0] for c in cfg.FOREIGN_CITIES}),
            "high_risk_category": int(txn["merchant_category"] in cfg.HIGH_RISK_CATEGORIES),
            "txn_count_5min": float(c5),
            "txn_count_1h": float(c1h),
            "sec_since_last": float(sec_since_last),
            "km_per_hour": 0.0 if math.isinf(km_per_hour) else float(km_per_hour),
        }
        return feats

    # ---------- escrita ----------

    def update(self, txn: dict) -> None:
        """Incorpora a transacao ao estado (chamar DEPOIS de online_features)."""
        ts = pd.to_datetime(txn["timestamp"])
        epoch = ts.timestamp()
        amount = float(txn["amount"])
        lat, lon = float(txn["lat"]), float(txn["lon"])

        # Welford
        self.n += 1
        delta = amount - self._mean
        self._mean += delta / self.n
        self._m2 += delta * (amount - self._mean)

        self.devices.add(txn["device_id"])
        self.coords.append((lat, lon))

        self.recent_ts.append(epoch)
        cutoff = epoch - WINDOW_SECONDS
        while self.recent_ts and self.recent_ts[0] < cutoff:
            self.recent_ts.popleft()

        self.last_ts, self.last_lat, self.last_lon = epoch, lat, lon

    # ---------- serializacao (ponto de troca para Redis) ----------

    def to_dict(self) -> dict:
        return {
            "customer_id": self.customer_id, "n": self.n,
            "mean": self._mean, "m2": self._m2,
            "devices": list(self.devices),
            "coords": list(self.coords),
            "recent_ts": list(self.recent_ts),
            "last_ts": self.last_ts, "last_lat": self.last_lat, "last_lon": self.last_lon,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CustomerState":
        st = cls(customer_id=int(d["customer_id"]))
        st.n, st._mean, st._m2 = d["n"], d["mean"], d["m2"]
        st.devices = set(d.get("devices", []))
        st.coords = deque(map(tuple, d.get("coords", [])), maxlen=MAX_COORDS)
        st.recent_ts = deque(d.get("recent_ts", []))
        st.last_ts, st.last_lat, st.last_lon = d.get("last_ts"), d.get("last_lat"), d.get("last_lon")
        return st


class FeatureStore:
    """Registro de estado por cliente. Trocavel por Redis (mesma interface)."""

    def __init__(self):
        self._states: dict[int, CustomerState] = {}

    def state(self, customer_id: int) -> CustomerState:
        cid = int(customer_id)
        if cid not in self._states:
            self._states[cid] = CustomerState(customer_id=cid)
        return self._states[cid]

    def snapshot(self, customer_id: int) -> dict:
        return self.state(customer_id).snapshot()

    def score_features(self, txn: dict) -> dict:
        """FEATURE_COLS da transacao contra o estado atual do cliente."""
        return self.state(txn["customer_id"]).online_features(txn)

    def update(self, txn: dict) -> None:
        self.state(txn["customer_id"]).update(txn)

    def known(self, customer_id: int) -> bool:
        return int(customer_id) in self._states

    # ---------- bootstrap a partir de historico ----------

    def bootstrap(self, feats: pd.DataFrame) -> "FeatureStore":
        """
        Aquece o store com um historico ja conhecido (CSV de features/scored).
        Processa em ordem cronologica para que o estado fique consistente.
        """
        cols = ["timestamp", "customer_id", "amount", "device_id",
                "city", "merchant_category", "lat", "lon"]
        df = feats[cols].sort_values("timestamp")
        for row in df.itertuples(index=False):
            self.update({
                "timestamp": row.timestamp, "customer_id": row.customer_id,
                "amount": row.amount, "device_id": row.device_id,
                "city": row.city, "merchant_category": row.merchant_category,
                "lat": row.lat, "lon": row.lon,
            })
        return self


def features_row(feats: dict) -> pd.DataFrame:
    """Empacota o dict de features numa linha DataFrame na ordem do modelo."""
    return pd.DataFrame([feats])[FEATURE_COLS]
