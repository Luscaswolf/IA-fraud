"""
Renderizacao do relatorio de investigacao (texto e/ou rich).
"""
from __future__ import annotations

from .investigator import Investigation

_CHAIN = [
    "customer_history", "device", "location",
    "merchant", "behavior", "similar_cases",
]
_CHAIN_LABEL = {
    "customer_history": "Customer history",
    "device": "Device",
    "location": "Location",
    "merchant": "Merchant",
    "behavior": "Behavior pattern",
    "similar_cases": "Similar fraud cases",
}


def as_dict(inv: Investigation, true_pattern: str | None = None) -> dict:
    return {
        "transaction_id": inv.transaction_id,
        "customer_id": inv.customer_id,
        "ml_risk": round(inv.ml_risk, 4),
        "agent_score": round(inv.agent_score, 4),
        "fraud_probability": inv.fraud_probability,
        "inferred_pattern": inv.inferred_pattern,
        "true_pattern": true_pattern,
        "correct_motive": (true_pattern == inv.inferred_pattern
                           if true_pattern else None),
        "evidences": [
            {"step": e.step, "pattern": e.pattern,
             "signal": round(e.signal, 3), "text": e.text}
            for e in inv.top_evidences()
        ],
        "similar_cases": [
            {"transaction_id": int(tid), "pattern": pat, "distance": round(d, 3)}
            for tid, pat, d in inv.similar_cases
        ],
    }


def render_text(inv: Investigation, true_pattern: str | None = None) -> str:
    prob = inv.fraud_probability
    lines = []
    lines.append("=" * 60)
    lines.append(f"  INVESTIGACAO - Transaction #{inv.transaction_id}")
    lines.append("=" * 60)
    lines.append(f"Cliente:              #{inv.customer_id}")
    lines.append(f"Risco do modelo (ML): {inv.ml_risk:.0%}")
    lines.append(f"Score do agente:      {inv.agent_score:.0%}")
    lines.append("")
    lines.append(f">> Fraud probability: {prob:.0%}")
    lines.append(f">> Motivo provavel:   {inv.inferred_pattern}")
    if true_pattern:
        ok = "CORRETO" if true_pattern == inv.inferred_pattern else "DIVERGENTE"
        lines.append(f">> Motivo real:       {true_pattern}  [{ok}]")
    lines.append("")
    lines.append("Cadeia de investigacao / evidencias:")
    lines.append("-" * 60)

    by_step = {}
    for e in inv.evidences:
        by_step.setdefault(e.step, []).append(e)
    for step in _CHAIN:
        if step in by_step:
            lines.append(f"  [{_CHAIN_LABEL[step]}]")
            for e in sorted(by_step[step], key=lambda x: -x.contribution):
                lines.append(f"     - {e.text}  (sinal {e.signal:.0%})")
    if inv.similar_cases:
        ids = ", ".join(f"#{t}" for t, _, _ in inv.similar_cases[:5])
        lines.append(f"  [Casos similares mais proximos] {ids}")
    lines.append("=" * 60)
    return "\n".join(lines)


def render_rich(inv: Investigation, true_pattern: str | None = None):
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        print(render_text(inv, true_pattern))
        return

    console = Console()
    prob = inv.fraud_probability
    color = "red" if prob >= 0.7 else "yellow" if prob >= 0.4 else "green"

    header = (
        f"[bold]Transaction #{inv.transaction_id}[/bold]   "
        f"Cliente #{inv.customer_id}\n"
        f"Risco ML: {inv.ml_risk:.0%}    Agente: {inv.agent_score:.0%}\n"
        f"[bold {color}]Fraud probability: {prob:.0%}[/bold {color}]\n"
        f"Motivo provavel: [bold]{inv.inferred_pattern}[/bold]"
    )
    if true_pattern:
        ok = true_pattern == inv.inferred_pattern
        tag = "[green]CORRETO[/green]" if ok else "[red]DIVERGENTE[/red]"
        header += f"\nMotivo real: {true_pattern}  {tag}"
    console.print(Panel(header, title="AI Fraud Investigator", border_style=color))

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Etapa")
    table.add_column("Sinal", justify="right")
    table.add_column("Evidencia")
    by_step = {}
    for e in inv.evidences:
        by_step.setdefault(e.step, []).append(e)
    for step in _CHAIN:
        for e in sorted(by_step.get(step, []), key=lambda x: -x.contribution):
            table.add_row(_CHAIN_LABEL[step], f"{e.signal:.0%}", e.text)
    console.print(table)

    if inv.similar_cases:
        ids = ", ".join(f"#{t} ({p})" for t, p, _ in inv.similar_cases[:5])
        console.print(f"[dim]Casos de fraude similares: {ids}[/dim]")
