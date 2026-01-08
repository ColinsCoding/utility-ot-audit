import typer
from rich.console import Console
from pathlib import Path

from odat2.audit_engine import AuditEngine
from odat2.reports.json_report import JSONReporter
from odat2.reports.html_report import HTMLReporter

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    csv_file: str = typer.Argument(..., help="Path to CSV file to audit"),
    out_json: str = typer.Option(None, "--out-json", help="Output JSON report path"),
    out_html: str = typer.Option(None, "--out-html", help="Output HTML report path"),
):
    """Audit a cable export CSV and produce issues + optional reports."""
    csv_path = Path(csv_file)
    if not csv_path.exists():
        raise typer.BadParameter(f"CSV file not found: {csv_path}")

    engine = AuditEngine()
    issues = engine.audit(str(csv_path))

    console.print(f"\n[bold]Audit complete[/bold] — {len(issues)} issue(s) found.")

    # Default report names if user asked for format but didn't supply explicit path
    if out_json is None and out_html is None:
        # No report requested; still show sample issues below
        pass

    if out_json:
        JSONReporter().generate(issues, out_json)
        console.print(f"[green]✓[/green] JSON report saved to: {out_json}")

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
