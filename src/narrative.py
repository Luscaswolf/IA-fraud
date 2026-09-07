"""
Camada de narrativa em linguagem natural.

Transforma a investigacao estruturada em um laudo textual, como um analista
escreveria. Funciona em dois modos:

  1. LLM real  -> se houver OPENAI_API_KEY ou ANTHROPIC_API_KEY no ambiente,
                  usa o modelo para redigir o laudo a partir das evidencias.
  2. Fallback  -> narrativa deterministica baseada em template (sem custo,
                  sem dependencia externa). Sempre disponivel.

O prompt recebe SEMPRE as evidencias estruturadas (report.as_dict), entao o
LLM nunca "inventa" fatos: ele apenas redige o que o agente ja apurou.
"""
from __future__ import annotations

import json
import os

from .investigator import Investigation
from .report import as_dict

_PATTERN_PT = {
    "high_amount": "valor atipico",
    "new_device": "novo dispositivo",
    "impossible_travel": "deslocamento impossivel",
    "velocity_burst": "rajada de transacoes",
    "card_testing": "teste de cartao",
    "account_takeover": "tomada de conta (account takeover)",
    "risky_merchant": "estabelecimento de alto risco",
    "indeterminado": "indeterminado",
}

_SYSTEM = (
    "Voce e um analista senior de fraude. Escreva um laudo objetivo em "
    "portugues do Brasil baseado EXCLUSIVAMENTE nas evidencias fornecidas em "
    "JSON. Nao invente dados. Estrutura: (1) veredito e probabilidade; "
    "(2) motivo principal; (3) lista das evidencias em bullets; "
    "(4) recomendacao de acao (aprovar / revisar / bloquear). "
    "Seja conciso (maximo ~150 palavras)."
)


def _recommendation(prob: float) -> str:
    # limiares calibrados por custo (src/calibration.py); fallback 0.4 / 0.7
    from .calibration import load_thresholds
    th = load_thresholds()
    if prob >= th["block"]:
        return "BLOQUEAR a transacao e acionar o cliente."
    if prob >= th["review"]:
        return "REVISAR manualmente antes de liberar."
    return "APROVAR com monitoramento padrao."


def _template_narrative(payload: dict) -> str:
    prob = payload["fraud_probability"]
    motive = _PATTERN_PT.get(payload["inferred_pattern"], payload["inferred_pattern"])
    veredito = ("ALTA suspeita de fraude" if prob >= 0.7
                else "suspeita MODERADA" if prob >= 0.4
                else "risco BAIXO")
    bullets = "\n".join(f"  - {e['text']}" for e in payload["evidences"])
    lines = [
        f"LAUDO DE INVESTIGACAO - Transaction #{payload['transaction_id']} "
        f"(Cliente #{payload['customer_id']})",
        "",
        f"Veredito: {veredito} (probabilidade {prob:.0%}).",
        f"Motivo principal identificado: {motive}.",
        "",
        "Evidencias apuradas:",
        bullets,
        "",
        f"Recomendacao: {_recommendation(prob)}",
    ]
    return "\n".join(lines)


def _llm_narrative(payload: dict) -> str | None:
    """Tenta usar um LLM real. Retorna None se indisponivel."""
    prompt = (
        "Evidencias da investigacao (JSON):\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    # --- OpenAI ---
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            print(f"[narrative] OpenAI indisponivel: {exc}")

    # --- Anthropic ---
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            import anthropic
            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
                max_tokens=400,
                system=_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as exc:  # pragma: no cover
            print(f"[narrative] Anthropic indisponivel: {exc}")

    return None


def narrate(inv: Investigation, true_pattern: str | None = None,
            use_llm: bool = True) -> dict:
    """Retorna dict com o laudo estruturado + narrativa em texto."""
    payload = as_dict(inv, true_pattern)
    narrative = None
    engine = "template"
    if use_llm:
        narrative = _llm_narrative(payload)
        if narrative:
            engine = "llm"
    if narrative is None:
        narrative = _template_narrative(payload)
    payload["narrative"] = narrative
    payload["narrative_engine"] = engine
    payload["recommendation"] = _recommendation(payload["fraud_probability"])
    return payload
