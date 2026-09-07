"""
API + Front-end do AI Fraud Investigator.

Sobe um servico FastAPI que:
  - serve a interface web (frontend/index.html) em  /
  - investiga transacoes do dataset de exemplo
  - permite ENVIAR UM CSV proprio e investiga-lo

Uso:
  python api.py            (ou: uvicorn api:app --reload)
  Abra:  http://127.0.0.1:8000

Endpoints:
  GET  /health
  GET  /api/datasets                              -> datasets disponiveis
  GET  /api/datasets/{ds}/top?n=20                -> maiores riscos
  GET  /api/datasets/{ds}/investigate/{tid}       -> laudo + narrativa
  POST /api/upload   (multipart: file=CSV)        -> cria dataset novo
  GET  /api/template.csv                          -> modelo de CSV
"""
from __future__ import annotations

import io
import uuid
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from src import config as cfg
from src import model as model_mod
from src.features import REQUIRED_COLS, build_features
from src.investigator import FraudInvestigator
from src.narrative import narrate

SCORED_CSV = cfg.DATA_DIR / "transactions_scored.csv"
FRONTEND_DIR = Path(__file__).parent / "frontend"

app = FastAPI(title="AI Fraud Investigator", version="2.0.0")

# ---- registro de datasets em memoria ----
# cada dataset: {"name": str, "feats": DataFrame, "engine": FraudInvestigator}
DATASETS: dict[str, dict] = {}


@lru_cache(maxsize=1)
def _model():
    if not cfg.MODEL_PATH.exists():
        raise RuntimeError("Rode 'python main.py train' antes de subir a API.")
    return joblib.load(cfg.MODEL_PATH)


def _fraud_library() -> pd.DataFrame | None:
    """Casos de fraude conhecidos do dataset de exemplo (referencia)."""
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


# nomes amigaveis para datasets reais baixados (arquivos data/*_scored.csv)
REAL_NAMES = {
    "real_creditcard": "Cartao de credito - dados reais (HuggingFace)",
    "real_nigerian": "Transacoes financeiras - dados reais (HuggingFace)",
}


def _load_default():
    """Carrega o dataset de exemplo e os datasets reais baixados."""
    if "default" in DATASETS:
        return
    if SCORED_CSV.exists():
        feats = pd.read_csv(SCORED_CSV)
        _register("Dataset de exemplo (100k transacoes sinteticas)", feats, ds_id="default")
    # datasets reais: qualquer data/real_*_scored.csv
    for path in sorted(cfg.DATA_DIR.glob("real_*_scored.csv")):
        key = path.stem.replace("_scored", "")
        try:
            feats = pd.read_csv(path)
            _register(REAL_NAMES.get(key, key), feats, ds_id=key)
        except Exception as exc:
            print(f"[warn] falha ao carregar {path.name}: {exc}")


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


# ------------------------- endpoints de API -------------------------

@app.get("/health")
def health():
    _load_default()
    return {"status": "ok", "datasets": list(DATASETS.keys())}


@app.get("/api/datasets")
def list_datasets():
    _load_default()
    return [
        {"id": k, "name": v["name"], "transactions": int(len(v["feats"]))}
        for k, v in DATASETS.items()
    ]


@app.get("/api/datasets/{ds}/stats")
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


@app.get("/api/datasets/{ds}/top")
def top(ds: str, n: int = Query(20, ge=1, le=200)):
    d = _get(ds)
    feats = d["feats"]
    cols = ["transaction_id", "customer_id", "amount", "merchant_category", "risk"]
    for opt in ("is_fraud", "fraud_pattern"):
        if opt in feats.columns:
            cols.append(opt)
    top_df = feats.sort_values("risk", ascending=False).head(n)[cols]
    return top_df.to_dict("records")


@app.get("/api/datasets/{ds}/investigate/{tid}")
def investigate(ds: str, tid: int, use_llm: bool = True):
    return _investigate(ds, tid, use_llm)


@app.post("/api/upload")
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
            f"Colunas obrigatorias ausentes: {missing}. "
            f"Esperado: {REQUIRED_COLS}",
        )
    try:
        feats = build_features(df)                 # deriva perfis do proprio CSV
        feats["risk"] = model_mod.score(_model(), feats)
    except Exception as exc:
        raise HTTPException(400, f"Falha ao processar: {exc}")

    ds_id = _register(f"Upload: {file.filename}", feats, use_reference=True)
    n_high = int((feats["risk"] >= 0.7).sum())
    return {
        "dataset_id": ds_id,
        "name": file.filename,
        "transactions": int(len(feats)),
        "high_risk": n_high,
    }


@app.get("/api/template.csv")
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
