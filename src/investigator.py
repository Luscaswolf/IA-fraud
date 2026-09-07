"""
Agente Investigador de Fraude.

Diferente de um classificador que so devolve um numero, este agente
PERCORRE UMA CADEIA DE INVESTIGACAO para uma transacao sinalizada:

    Transaction -> Customer history -> Previous transactions -> Device
    -> Location -> Merchant -> Behavior pattern -> Similar fraud cases

Cada etapa produz EVIDENCIAS. O agente pondera as evidencias, infere o
provavel MOTIVO da fraude e monta um relatorio explicavel.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from . import config as cfg
from .features import FEATURE_COLS


@dataclass
class Evidence:
    step: str          # etapa da investigacao (device, location, ...)
    signal: float      # 0..1 forca do indicio
    weight: float      # peso relativo do indicio
    pattern: str       # padrao de fraude que este indicio sugere
    text: str          # explicacao legivel

    @property
    def contribution(self) -> float:
        return self.signal * self.weight


@dataclass
class Investigation:
    transaction_id: int
    customer_id: int
    ml_risk: float
    evidences: list = field(default_factory=list)
    similar_cases: list = field(default_factory=list)

    def add(self, ev: Evidence):
        if ev.signal > 0:
            self.evidences.append(ev)

    @property
    def agent_score(self) -> float:
        """Score do agente derivado das evidencias (0..1)."""
        if not self.evidences:
            return 0.0
        total_w = sum(e.weight for e in self.evidences)
        raw = sum(e.contribution for e in self.evidences) / max(total_w, 1e-9)
        # boost por acumulo de indicios independentes
        n_strong = sum(1 for e in self.evidences if e.signal >= 0.5)
        boost = min(0.25, 0.06 * max(0, n_strong - 1))
        return float(min(1.0, raw + boost))

    @property
    def fraud_probability(self) -> float:
        """Combina o modelo ML com o raciocinio do agente."""
        return round(0.5 * self.ml_risk + 0.5 * self.agent_score, 4)

    @property
    def inferred_pattern(self) -> str:
        """Motivo mais provavel = padrao com maior contribuicao acumulada."""
        if not self.evidences:
            return "indeterminado"
        acc = {}
        for e in self.evidences:
            acc[e.pattern] = acc.get(e.pattern, 0) + e.contribution
        return max(acc, key=acc.get)

    def top_evidences(self, k=6):
        return sorted(self.evidences, key=lambda e: -e.contribution)[:k]


class FraudInvestigator:
    """Orquestra a cadeia de investigacao sobre a base de features."""

    def __init__(self, feats: pd.DataFrame, fraud_reference: pd.DataFrame | None = None):
        self.feats = feats.set_index("transaction_id", drop=False)
        self.by_customer = {
            cid: g.sort_values("timestamp")
            for cid, g in self.feats.groupby("customer_id")
        }
        # base de casos de fraude conhecidos para busca de similares:
        # usa uma referencia externa (biblioteca de fraudes) se fornecida,
        # senao os rotulos do proprio dataset (se existirem).
        self._fit_similarity(fraud_reference)

    def _fit_similarity(self, fraud_reference=None):
        if fraud_reference is not None:
            frauds = fraud_reference
        elif "is_fraud" in self.feats.columns:
            frauds = self.feats[self.feats["is_fraud"] == 1]
        else:
            frauds = self.feats.iloc[0:0]  # vazio -> sem casos similares
        self._fraud_ids = frauds["transaction_id"].values
        self._fraud_patterns = (frauds["fraud_pattern"].values
                                if "fraud_pattern" in frauds.columns
                                else np.array(["desconhecido"] * len(frauds)))
        X = frauds[FEATURE_COLS].fillna(0).values
        if len(X) > 1:
            self._scaler_mean = X.mean(axis=0)
            self._scaler_std = X.std(axis=0) + 1e-9
            Xn = (X - self._scaler_mean) / self._scaler_std
            self._nn = NearestNeighbors(n_neighbors=min(6, len(X))).fit(Xn)
            self._fraud_X = Xn
        else:
            self._nn = None

    # ---------- etapas da cadeia de investigacao ----------

    def investigate(self, transaction_id: int) -> Investigation:
        if transaction_id not in self.feats.index:
            raise KeyError(f"Transacao {transaction_id} nao encontrada")
        row = self.feats.loc[transaction_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        inv = Investigation(
            transaction_id=int(row["transaction_id"]),
            customer_id=int(row["customer_id"]),
            ml_risk=float(row.get("risk", 0.0)),
        )

        hist = self.by_customer[row["customer_id"]]
        past = hist[hist["timestamp"] < row["timestamp"]]

        self._step_amount(inv, row)
        self._step_device(inv, row, past)
        self._step_location(inv, row, past)
        self._step_velocity(inv, row)
        self._step_merchant(inv, row)
        self._step_behavior(inv, row, past)
        self._step_similar_cases(inv, row)
        return inv

    def _step_amount(self, inv, row):
        ratio = float(row["amount_ratio"])
        if ratio >= 3:
            signal = min(1.0, (ratio - 1) / 8)
            inv.add(Evidence(
                "customer_history", signal, 1.0, "high_amount",
                f"Valor R$ {row['amount']:.2f} e {ratio:.1f}x superior "
                f"a media do cliente (R$ {row['avg_amount']:.2f}).",
            ))

    def _step_device(self, inv, row, past):
        if int(row["is_new_device"]) == 1:
            seen = past["device_id"].nunique() if len(past) else 0
            inv.add(Evidence(
                "device", 0.8, 0.9, "new_device",
                f"Dispositivo '{row['device_id']}' nunca utilizado antes "
                f"(cliente possui {seen} dispositivo(s) no historico).",
            ))

    def _step_location(self, inv, row, past):
        dist = float(row["dist_from_home_km"])
        if int(row["is_foreign_city"]) == 1:
            inv.add(Evidence(
                "location", 0.85, 0.9, "impossible_travel",
                f"Transacao em cidade estrangeira: {row['city']} "
                f"({dist:,.0f} km da residencia).",
            ))
        elif dist > 400:
            inv.add(Evidence(
                "location", min(1.0, dist / 2000), 0.6, "impossible_travel",
                f"Localizacao incomum: {row['city']} ({dist:,.0f} km de casa).",
            ))
        # impossible travel real: velocidade geografica somente quando
        # ha deslocamento fisico relevante (evita ruido de micro-jitter local)
        kmh = float(row["km_per_hour"])
        if kmh > 900 and dist > 200:  # mais rapido que aviao comercial
            inv.add(Evidence(
                "location", 0.95, 1.0, "impossible_travel",
                f"Deslocamento fisicamente impossivel: {kmh:,.0f} km/h "
                f"desde a transacao anterior.",
            ))

    def _step_velocity(self, inv, row):
        c5 = float(row["txn_count_5min"])
        sec = float(row["sec_since_last"])
        amount = float(row["amount"])
        is_micro = amount < 5
        if c5 >= 3 and is_micro:
            # rajada de micro-valores => teste de cartao
            inv.add(Evidence(
                "behavior", min(1.0, c5 / 8 + 0.2), 1.0, "card_testing",
                f"Padrao de teste de cartao: {int(c5)} micro-transacoes "
                f"(R$ {amount:.2f}) em sequencia (ultimo intervalo {sec:.0f}s).",
            ))
        elif c5 >= 4:
            inv.add(Evidence(
                "behavior", min(1.0, c5 / 8), 0.9, "velocity_burst",
                f"{int(c5)} transacoes em menos de 5 minutos "
                f"(intervalo desde a anterior: {sec:.0f}s).",
            ))

    def _step_merchant(self, inv, row):
        if int(row["high_risk_category"]) == 1:
            night = int(row["is_night"]) == 1
            signal = 0.7 if night else 0.4
            when = " em horario atipico (madrugada)" if night else ""
            inv.add(Evidence(
                "merchant", signal, 0.6, "risky_merchant",
                f"Categoria de alto risco: {row['merchant_category']}{when}.",
            ))

    def _step_behavior(self, inv, row, past):
        # combinacao classica de account takeover
        if int(row["is_new_device"]) and int(row["is_foreign_city"]) \
                and float(row["amount_ratio"]) >= 3:
            inv.add(Evidence(
                "behavior", 0.95, 1.1, "account_takeover",
                "Combinacao critica: novo dispositivo + nova localizacao "
                "+ valor muito acima do padrao (sinal de tomada de conta).",
            ))
        # mudanca abrupta de comportamento
        if len(past) >= 5:
            zs = float(row["amount_zscore"])
            if zs > 4:
                inv.add(Evidence(
                    "behavior", min(1.0, zs / 10), 0.5, "high_amount",
                    f"Desvio de {zs:.1f} sigmas em relacao ao historico "
                    f"({len(past)} transacoes anteriores analisadas).",
                ))

    def _step_similar_cases(self, inv, row):
        if self._nn is None:
            return
        x = pd.to_numeric(row[FEATURE_COLS], errors="coerce").fillna(0).values.astype(float)
        xn = ((x - self._scaler_mean) / self._scaler_std).reshape(1, -1)
        dist, idx = self._nn.kneighbors(xn)
        dist, idx = dist[0], idx[0]
        # similaridade -> quantos casos "proximos"
        close = [(self._fraud_ids[i], self._fraud_patterns[i], float(d))
                 for i, d in zip(idx, dist)
                 if self._fraud_ids[i] != row["transaction_id"] and d < 3.0]
        inv.similar_cases = close
        if close:
            # padrao dominante entre os vizinhos
            pats = [p for _, p, _ in close]
            dom = max(set(pats), key=pats.count)
            n_total_similar = int((self._pairwise_close(xn) if len(self._fraud_X) else 0))
            inv.add(Evidence(
                "similar_cases", min(1.0, len(close) / 5), 0.8, dom,
                f"Perfil semelhante a {n_total_similar} casos de fraude "
                f"conhecidos (padrao dominante: {dom}).",
            ))

    def _pairwise_close(self, xn, radius=2.5):
        d = np.linalg.norm(self._fraud_X - xn, axis=1)
        return int((d < radius).sum())
