"""
API + Front-end do AI Fraud Investigator.

Sobe um servico FastAPI que:
  - serve a interface web (frontend/index.html) em  /
  - investiga transacoes do dataset de exemplo
  - permite ENVIAR UM CSV proprio e investiga-lo
  - PONTUA UMA transacao em tempo real (POST /score) via feature store online
  - recebe FEEDBACK do analista (POST /api/feedback) para re-treino

Uso:
  python api.py            (ou: uvicorn api:app --reload)
  Abra:  http://127.0.0.1:8000

Variaveis de ambiente:
  FRAUD_API_KEY      se definida, exige header  x-api-key  nas rotas /api e /score
  FRAUD_RATE_LIMIT   requisicoes por minuto por cliente (default 120)

Endpoints:
  GET  /health
  GET  /api/datasets                              -> datasets disponiveis
  GET  /api/datasets/{ds}/top?n=20                -> maiores riscos
  GET  /api/datasets/{ds}/investigate/{tid}       -> laudo + narrativa
  POST /api/upload   (multipart: file=CSV)        -> cria dataset novo (persistido)
  POST /api/feedback  {transaction_id,label}      -> grava decisao do analista
  POST /score         {transacao}                 -> decisao em tempo real
  GET  /api/template.csv                          -> modelo de CSV
"""
from __future__ import annotations

import io
import json
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import (Depends, FastAPI, File, Header, HTTPException, Query, Request,
                     UploadFile)
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import calibration
from src import config as cfg
from src import feedback as feedback_mod
from src import model as model_mod
from src.feature_store import FeatureStore, features_row
from src.features import REQUIRED_COLS, build_features
from src.investigator import FraudInvestigator
from src.narrative import narrate

SCORED_CSV = cfg.DATA_DIR / "transactions_scored.csv"
FRONTEND_DIR = Path(__file__).parent / "frontend"
UPLOAD_DIR = cfg.DATA_DIR / "uploads"
UPLOAD_INDEX = UPLOAD_DIR / "index.json"

API_KEY = os.getenv("FRAUD_API_KEY")
RATE_LIMIT = int(os.getenv("FRAUD_RATE_LIMIT", "120"))

app = FastAPI(title="AI Fraud Investigator", version="3.0.0")

# ---- registro de datasets em memoria ----
DATASETS: dict[str, dict] = {}
# ---- feature store online (item 2) ----
STORE = FeatureStore()
_STORE_READY = False


@lru_cache(maxsize=1)
def _model():
    if not cfg.MODEL_PATH.exists():
        raise RuntimeError("Rode 'python main.py train' antes de subir a API.")
    return joblib.load(cfg.MODEL_PATH)


# ------------------------- seguranca: api key + rate limit -------------------------

def require_api_key(x_api_key: str | None = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(401, "x-api-key ausente ou invalida")


_HITS: dict[str, deque] = defaultdict(deque)


def rate_limit(request: Request):
    if RATE_LIMIT <= 0:
        return
    ident = request.headers.get("x-api-key") or (request.client.host if request.client else "?")
    now = time.time()
    q = _HITS[ident]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        raise HTTPException(429, f"limite de {RATE_LIMIT} req/min excedido")
    q.append(now)


GUARDED = [Depends(rate_limit), Depends(require_api_key)]


# ------------------------- datasets -------------------------

def _fraud_library() -> pd.DataFrame | None:
    _load_default()
    d = DATASETS.get("default")
    if d is None or "is_fraud" not in d["feats"].columns:
        return None
    return d["feats"][d["feats"]["is_fraud"] == 1]


def _register(name: str, feats: pd.DataFrame, ds_id: str | None = None,
              use_reference: bool = False) -> str:
    ds_id = ds_id or uuid.uuid4().hex[:8]
    ref = _fraud_library() if use_reference else None
    DATASETS[ds_id] = {
        "name": name,
        "feats": feats.reset_index(drop=True),
        "engine": FraudInvestigator(feats, fraud_reference=ref),
    }
    return ds_id


REAL_NAMES = {
    "real_creditcard": "Cartao de credito - dados reais (HuggingFace)",
    "real_nigerian": "Transacoes financeiras - dados reais (HuggingFace)",
}


def _persist_upload(ds_id: str, name: str, feats: pd.DataFrame) -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    feats.to_csv(UPLOAD_DIR / f"{ds_id}.csv", index=False)
    idx = json.loads(UPLOAD_INDEX.read_text()) if UPLOAD_INDEX.exists() else {}
    idx[ds_id] = {"name": name, "saved_at": datetime.now(timezone.utc).isoformat()}
    UPLOAD_INDEX.write_text(json.dumps(idx, indent=2))


def _load_default():
    global _STORE_READY
    if "default" in DATASETS:
        return
    if SCORED_CSV.exists():
        feats = pd.read_csv(SCORED_CSV)
        _register("Dataset de exemplo (100k transacoes sinteticas)", feats, ds_id="default")
        if not _STORE_READY:
            STORE.bootstrap(feats)
            _STORE_READY = True
    for path in sorted(cfg.DATA_DIR.glob("real_*_scored.csv")):
        key = path.stem.replace("_scored", "")
        try:
            _register(REAL_NAMES.get(key, key), pd.read_csv(path), ds_id=key)
        except Exception as exc:
            print(f"[warn] falha ao carregar {path.name}: {exc}")
    # uploads persistidos entre reinicios (item 7)
    if UPLOAD_INDEX.exists():
        idx = json.loads(UPLOAD_INDEX.read_text())
        for ds_id, meta in idx.items():
            fpath = UPLOAD_DIR / f"{ds_id}.csv"
            if fpath.exists():
                try:
                    _register(f"Upload: {meta['name']}", pd.read_csv(fpath),
                              ds_id=ds_id, use_reference=True)
                except Exception as exc:
                    print(f"[warn] falha ao recarregar upload {ds_id}: {exc}")


def _get(ds: str) -> dict:
    _load_default()
    if ds not in DATASETS:
        raise HTTPException(404, f"Dataset '{ds}' nao encontrado")
    return DATASETS[ds]


def _investigate(ds: str, tid: int, use_llm: bool) -> dict:
    d = _get(ds)
    feats = d["feats"]
    matches = feats[feats.transaction_id == tid]
    if matches.empty:
        raise HTTPException(404, f"Transacao {tid} nao encontrada")
    row = matches.iloc[0]
    inv = d["engine"].investigate(tid)
    true = row["fraud_pattern"] if "fraud_pattern" in row and row.get("is_fraud", 0) == 1 else None
    return narrate(inv, true, use_llm=use_llm)


# ------------------------- endpoints -------------------------

@app.get("/health")
def health():
    _load_default()
    return {"status": "ok", "datasets": list(DATASETS.keys()),
            "auth_required": bool(API_KEY), "store_customers": len(STORE._states)}


@app.get("/api/datasets", dependencies=GUARDED)
def list_datasets():
    _load_default()
    return [
        {"id": k, "name": v["name"], "transactions": int(len(v["feats"]))}
        for k, v in DATASETS.items()
    ]


@app.get("/api/datasets/{ds}/stats", dependencies=GUARDED)
def stats(ds: str):
    d = _get(ds)
    f = d["feats"]
    risk = f["risk"]
    high = f[risk >= 0.7]
    return {
        "total": int(len(f)),
        "high_risk": int((risk >= 0.7).sum()),
        "med_risk": int(((risk >= 0.4) & (risk < 0.7)).sum()),
        "avg_risk": float(risk.mean()),
        "exposed_amount": float(high["amount"].sum()),
        "labeled": bool("is_fraud" in f.columns),
        "known_frauds": int(f["is_fraud"].sum()) if "is_fraud" in f.columns else None,
    }


@app.get("/api/datasets/{ds}/top", dependencies=GUARDED)
def top(ds: str, n: int = Query(20, ge=1, le=200)):
    d = _get(ds)
    feats = d["feats"]
    cols = ["transaction_id", "customer_id", "amount", "merchant_category", "risk"]
    for opt in ("is_fraud", "fraud_pattern"):
        if opt in feats.columns:
            cols.append(opt)
    top_df = feats.sort_values("risk", ascending=False).head(n)[cols]
    return top_df.to_dict("records")


@app.get("/api/datasets/{ds}/investigate/{tid}", dependencies=GUARDED)
def investigate(ds: str, tid: int, use_llm: bool = True):
    return _investigate(ds, tid, use_llm)


@app.post("/api/upload", dependencies=GUARDED)
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"CSV invalido: {exc}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"Colunas obrigatorias ausentes: {missing}. Esperado: {REQUIRED_COLS}",
        )
    try:
        feats = build_features(df)
        feats["risk"] = model_mod.score(_model(), feats)
    except Exception as exc:
        raise HTTPException(400, f"Falha ao processar: {exc}")

    ds_id = _register(f"Upload: {file.filename}", feats, use_reference=True)
    try:
        _persist_upload(ds_id, file.filename or ds_id, feats)
    except Exception as exc:
        print(f"[warn] nao persistiu upload: {exc}")

    return {
        "dataset_id": ds_id,
        "name": file.filename,
        "transactions": int(len(feats)),
        "high_risk": int((feats["risk"] >= 0.7).sum()),
    }


class FeedbackIn(BaseModel):
    transaction_id: int
    label: str | int          # "fraud"/"legit" ou 1/0
    dataset_id: str = "default"
    analyst: str | None = None


@app.post("/api/feedback", dependencies=GUARDED)
def submit_feedback(fb: FeedbackIn):
    lbl = fb.label
    if isinstance(lbl, str):
        m = {"fraud": 1, "fraude": 1, "1": 1, "legit": 0, "legitima": 0, "0": 0}
        if lbl.lower() not in m:
            raise HTTPException(400, "label deve ser 'fraud' ou 'legit'")
        lbl = m[lbl.lower()]
    if lbl not in (0, 1):
        raise HTTPException(400, "label invalida")
    entry = feedback_mod.record(fb.transaction_id, int(lbl), fb.dataset_id, fb.analyst)
    total = len(feedback_mod.load())
    return {"recorded": entry, "total_feedback": total,
            "hint": "rode 'python main.py retrain' para realimentar o modelo"}


# ---- scoring em tempo real (itens 1 + 2) ----

class ScoreIn(BaseModel):
    customer_id: int
    amount: float
    merchant_category: str
    device_id: str
    city: str
    lat: float
    lon: float
    transaction_id: int | None = None
    timestamp: str | None = None
    update_state: bool = Field(default=True,
                               description="incorpora a transacao ao estado do cliente")


@lru_cache(maxsize=1)
def _online_engine() -> FraudInvestigator:
    _load_default()
    d = DATASETS["default"]
    return FraudInvestigator(d["feats"], fraud_reference=_fraud_library())


@app.post("/score", dependencies=GUARDED)
def score(inp: ScoreIn):
    t0 = time.perf_counter()
    _load_default()
    txn = {
        "transaction_id": inp.transaction_id or int(time.time() * 1000) % 2_000_000_000,
        "timestamp": inp.timestamp or datetime.now(timezone.utc).isoformat(),
        "customer_id": inp.customer_id,
        "amount": inp.amount,
        "merchant_category": inp.merchant_category,
        "device_id": inp.device_id,
        "city": inp.city,
        "lat": inp.lat,
        "lon": inp.lon,
    }

    st = STORE.state(inp.customer_id)
    known_before = st.n
    feats = st.online_features(txn)
    risk = float(model_mod.score(_model(), features_row(feats)).iloc[0])

    snap = st.snapshot()
    row = pd.Series({
        **feats,
        "transaction_id": txn["transaction_id"],
        "customer_id": inp.customer_id,
        "risk": risk,
        "avg_amount": snap["avg_amount"],
        "device_id": inp.device_id,
        "city": inp.city,
        "merchant_category": inp.merchant_category,
    })

    # historico minimo para as etapas que contam dispositivos / tamanho de historico
    dev = list(st.devices)
    pad = max(0, st.n - len(dev))
    past = pd.DataFrame({"device_id": dev + [None] * pad})

    inv = _online_engine().investigate_row(row, past)
    payload = narrate(inv, None, use_llm=False)

    th = calibration.load_thresholds()
    prob = payload["fraud_probability"]
    decision = ("bloquear" if prob >= th["block"]
                else "revisar" if prob >= th["review"] else "aprovar")

    if inp.update_state:
        STORE.update(txn)

    return {
        "transaction_id": txn["transaction_id"],
        "customer_id": inp.customer_id,
        "decision": decision,
        "thresholds": th,
        "ml_risk": round(risk, 4),
        "agent_score": payload["agent_score"],
        "fraud_probability": prob,
        "inferred_pattern": payload["inferred_pattern"],
        "recommendation": payload["recommendation"],
        "evidences": payload["evidences"],
        "customer_state": {**snap, "transactions_seen_before": known_before},
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


@app.get("/api/template.csv", dependencies=GUARDED)
def template():
    if SCORED_CSV.exists():
        sample = pd.read_csv(SCORED_CSV, nrows=8)[REQUIRED_COLS]
        return PlainTextResponse(sample.to_csv(index=False), media_type="text/csv")
    return PlainTextResponse(",".join(REQUIRED_COLS) + "\n", media_type="text/csv")


# ------------------------- front-end estatico -------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
