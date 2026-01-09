import json
import typer
from rich.console import Console
from odat2.validators.review_priority import compute_review_priority
from pathlib import Path

from odat2.audit_engine import AuditEngine
from odat2.reports.json_report import JSONReporter
from odat2.reports.html_report import HTMLReporter
from odat2.reports.terminal_report import print_terminal_summary

from datetime import date, datetime
from odat2.validators.doc_confidence import compute_doc_confidence

import pandas as pd

from odat2.validators.uncertainty import monte_carlo_review_priority

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    csv_file: str = typer.Argument(..., help="Path to CSV file to audit"),
    out_json: str = typer.Option(None, "--out-json", help="Output JSON report path"),
    out_html: str = typer.Option(None, "--out-html", help="Output HTML report path"),
    doc_half_life_days: float = typer.Option(180.0, "--doc-half-life-days", help="Documentation confidence half-life in days"),
    today: str = typer.Option("", "--today", help="Override today's date (YYYY-MM-DD) for reproducible runs"),

    rps_sigma: float = typer.Option(0.05, "--rps-sigma", help="Std dev for doc_confidence uncertainty"),
    rps_mc_n: int = typer.Option(5000, "--rps-mc-n", help="Monte Carlo samples for RPS uncertainty"),
    rps_seed: int = typer.Option(0, "--rps-seed", help="Random seed for RPS uncertainty"),
):


    """Audit a cable export CSV and produce issues + optional reports."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise typer.BadParameter(f"CSV file not found: {csv_path}")

    try:
        today_date = date.today() if not today else datetime.strptime(today, "%Y-%m-%d").date()
    except ValueError:
        raise typer.BadParameter("Invalid --today. Use YYYY-MM-DD (example: 2026-01-08)")

    engine = AuditEngine()
    issues = engine.audit(str(csv_path))

    errors = sum(1 for i in issues if i.severity == "error")
    warnings = sum(1 for i in issues if i.severity == "warning")

    type_counts = {}
    for i in issues:
        t = getattr(i, "issue_type", getattr(i, "type", "unknown"))
        type_counts[t] = type_counts.get(t, 0) + 1

    console.print(f"\n[bold]Audit complete[/bold] — {len(issues)} issue(s) found.")

    # Default report names if user asked for format but didn't supply explicit path
    if out_json is None and out_html is None:
        # No report requested; still show sample issues below
        pass

    df2 = pd.read_csv(csv_path)
    rows_processed = len(df2)

    last_verified = ""
    if "last_verified_date" in df2.columns:
        vals = [str(x).strip() for x in df2["last_verified_date"].dropna().tolist()]
        vals = [v for v in vals if v]
        if vals:
            last_verified = max(vals)  # latest YYYY-MM-DD

    doc = compute_doc_confidence(
        last_verified,
        today=today_date,
        half_life_days=doc_half_life_days,
    )

    rps = compute_review_priority(
        errors=errors,
        warnings=warnings,
        issue_type_counts=type_counts,
        doc_confidence=doc.confidence if doc.status != "UNKNOWN" else 0.0,
    )

    rps_u = monte_carlo_review_priority(
        errors=errors,
        warnings=warnings,
        issue_type_counts=type_counts,
        doc_confidence=doc.confidence if doc.status != "UNKNOWN" else 0.0,
        sigma_doc_confidence=rps_sigma,
        n=rps_mc_n,
        seed=rps_seed,
    )


    if out_json:
        JSONReporter().generate(issues, out_json)
        console.print(f"[green]OK[/green] JSON report saved to: {out_json}")

        p = Path(out_json)
        report = json.loads(p.read_text(encoding="utf-8"))

        report["doc_confidence"] = round(doc.confidence, 6)
        report["doc_status"] = doc.status
        report["doc_days_since_verified"] = doc.days_since_verified

        report["review_priority_score"] = rps.score_0_100
        report["review_priority"] = rps.level
        report["review_priority_drivers"] = rps.drivers

        report["review_priority_uncertainty"] = {
            "mean_score": rps_u.mean_score,
            "std_score": rps_u.std_score,
            "q05": rps_u.q05,
            "q50": rps_u.q50,
            "q95": rps_u.q95,
            "p_low": rps_u.p_low,
            "p_medium": rps_u.p_medium,
            "p_high": rps_u.p_high,
            "samples": rps_u.samples,
            "sigma_doc_confidence": rps_u.sigma_doc_confidence,
            "base_doc_confidence": rps_u.base_doc_confidence,
            "notes": rps_u.notes,
        }

        p.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print_terminal_summary(issues, rows_processed=rows_processed)
        console.print(f"Doc confidence: {doc.confidence:.2f} ({doc.status})")
        console.print(f"Review Priority: {rps.level} (RPS={rps.score_0_100:.2f})")


    if out_html:
        HTMLReporter().generate(issues, out_html)
        console.print(f"[green]✓[/green] HTML report saved to: {out_html}")

    if issues:
        console.print("\n[bold yellow]Sample issues:[/bold yellow]")
        for issue in issues[:10]:
            color = "red" if issue.severity == "error" else ("yellow" if issue.severity == "warning" else "cyan")
            console.print(f"[{color}]{issue.severity.upper()}[/{color}] {issue.issue_type}: {issue.message}")


if __name__ == "__main__":

    app()
