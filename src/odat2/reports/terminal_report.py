from __future__ import annotations

from collections import Counter
from rich.console import Console
from rich.table import Table


def print_terminal_summary(issues, rows_processed: int) -> None:
    """
    Prints a safe, deterministic summary table to the terminal.
    Expects items with attributes:
      - severity (e.g., 'error', 'warning', 'info')
      - issue_type (string category)
    """
    console = Console()

    sev = [getattr(i, "severity", "unknown") for i in issues]
    types = [getattr(i, "issue_type", "unknown") for i in issues]

    severity_counts = Counter(sev)
    type_counts = Counter(types)

    table = Table(title="ODAT Audit Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Rows processed", str(rows_processed))
    table.add_row("Errors", str(severity_counts.get("error", 0)))
    table.add_row("Warnings", str(severity_counts.get("warning", 0)))
    table.add_row("Info", str(severity_counts.get("info", 0)))
    table.add_row("Total issues", str(len(issues)))

    console.print(table)

    if type_counts:
        t = Table(title="Issues by Type")
        t.add_column("Type")
        t.add_column("Count", justify="right")
        for k, v in sorted(type_counts.items()):
            t.add_row(str(k), str(v))
        console.print(t)

    if severity_counts.get("error", 0) > 0:
        console.print("[bold red]❌ Audit FAILED — errors must be reviewed[/bold red]")
    elif severity_counts.get("warning", 0) > 0:
        console.print("[bold yellow]⚠️ Audit completed with warnings[/bold yellow]")
    else:
        console.print("[bold green]✅ Audit PASSED — no issues found[/bold green]")
