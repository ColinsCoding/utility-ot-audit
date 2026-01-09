import typer
from rich.console import Console
from pathlib import Path

from odat2.audit_engine import AuditEngine
from odat2.reports.json_report import JSONReporter
from odat2.reports.html_report import HTMLReporter
from odat2.reports.terminal_report import print_terminal_summary

from datetime import date, datetime
from odat2.validators.doc_confidence import compute_doc_confidence

import pandas as pd

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    csv_file: str = typer.Argument(..., help="Path to CSV file to audit"),
    out_json: str = typer.Option(None, "--out-json", help="Output JSON report path"),
    out_html: str = typer.Option(None, "--out-html", help="Output HTML report path"),
    doc_half_life_days: float = typer.Option(180.0, "--doc-half-life-days", help="Documentation confidence half-life in days"),
    today: str = typer.Option("", "--today", help="Override today's date (YYYY-MM-DD) for reproducible runs"),
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


    if out_json:
        JSONReporter().generate(issues, out_json)
        console.print(f"[green]OK[/green] JSON report saved to: {out_json}")
        
        import json
        p = Path(out_json)
        report = json.loads(p.read_text(encoding="utf-8"))
        report["doc_confidence"] = round(doc.confidence, 6)
        report["doc_status"] = doc.status
        report["doc_days_since_verified"] = doc.days_since_verified
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")



        df2 = pd.read_csv(csv_path)
        rows_processed = len(df2)

        last_verified = ""
        if "last_verified_date" in df2.columns:
            vals = [str(x).strip() for x in df2["last_verified_date"].dropna().tolist()]
            vals = [v for v in vals if v]
            if vals:
                # ISO dates compare correctly as strings; max = latest date
                last_verified = max(vals)

        doc = compute_doc_confidence(
            last_verified,
            today=today_date,
            half_life_days=doc_half_life_days,
        )

        print_terminal_summary(issues, rows_processed=rows_processed)
        console.print(f"Doc confidence: {doc.confidence:.2f} ({doc.status})")

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
