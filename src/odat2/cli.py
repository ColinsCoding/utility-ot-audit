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

from odat2.telecom.layout import load_layout
from odat2.telecom.route_optimizer import compute_routes
from odat2.telecom.dxf_writer import DXFPolyline, write_r12_dxf_polylines

from odat2.validators.uncertainty import monte_carlo_review_priority

try:
    from odat2.telecom.viz import plot_routes_preview
except ImportError:
    plot_routes_preview = None


from odat2.telecom.security.io import load_sensors_csv
from odat2.telecom.security.coverage import CoverageAnalyzer
from odat2.telecom.security.viz import plot_coverage



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




@app.command("route-optimize")
def route_optimize(
    layout_json: str = typer.Argument(..., help="Path to layout.json describing the routing grid (width/height/obstacles)."),
    endpoints_csv: str = typer.Argument(..., help="Path to endpoints.csv with from_x,from_y,to_x,to_y,cable_type,route_id."),
    out_dxf: str = typer.Option("routes.dxf", "--out-dxf", help="Output DXF drawing path (AutoCAD compatible)."),
    out_routes_csv: str = typer.Option("routes.csv", "--out-routes-csv", help="Per-route results CSV path."),
    out_bom_csv: str = typer.Option("bom.csv", "--out-bom-csv", help="BOM summary CSV path (length totals per cable type)."),
    grid_scale_m: float = typer.Option(1.0, "--grid-scale-m", help="Meters per grid cell for length calculations."),
    spare_pct: float = typer.Option(0.10, "--spare-pct", help="Spare cable percentage (e.g., 0.10 = +10%%)."),
):
    """
    Generate optimal telecom cable routes using A* and export an AutoCAD-compatible DXF.

    Workflow: layout.json + endpoints.csv → A* routing → routes.csv + bom.csv + routes.dxf
    """
    layout = load_layout(layout_json)
    df = pd.read_csv(endpoints_csv)

    required = {"from_x", "from_y", "to_x", "to_y"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise typer.BadParameter(f"endpoints_csv is missing required column(s): {', '.join(missing)}")

    # Fill optional columns
    if "route_id" not in df.columns:
        df["route_id"] = [f"route_{i+1}" for i in range(len(df))]
    if "cable_type" not in df.columns:
        df["cable_type"] = "unknown"

    results = compute_routes(layout, df.to_dict(orient="records"), grid_scale_m=grid_scale_m)

    # Write per-route CSV
    rows = []
    polylines = []
    ok_count = 0
    for r in results:
        rows.append({
            "route_id": r.route_id,
            "cable_type": r.cable_type,
            "from_x": r.start[0],
            "from_y": r.start[1],
            "to_x": r.goal[0],
            "to_y": r.goal[1],
            "status": r.status,
            "length_m": round(r.length, 3),
            "cost": round(r.cost, 3) if r.cost != float("inf") else "",
            "turns": r.turns,
            "message": r.message,
        })
        if r.status == "ok" and r.path:
            ok_count += 1
            layer = f"ROUTE_{r.cable_type}".replace(" ", "_").upper()
            pts = [(p[0] * grid_scale_m, p[1] * grid_scale_m) for p in r.path]
            polylines.append(DXFPolyline(layer=layer, points=pts, closed=False))

    pd.DataFrame(rows).to_csv(out_routes_csv, index=False)

    # BOM summary
    bom = {}
    for r in results:
        if r.status != "ok":
            continue
        bom.setdefault(r.cable_type, 0.0)
        bom[r.cable_type] += float(r.length)

    bom_rows = []
    for cable_type, total_len in sorted(bom.items(), key=lambda x: x[0]):
        with_spare = total_len * (1.0 + float(spare_pct))
        bom_rows.append({
            "cable_type": cable_type,
            "total_length_m": round(total_len, 3),
            "spare_pct": float(spare_pct),
            "length_with_spare_m": round(with_spare, 3),
        })
    pd.DataFrame(bom_rows).to_csv(out_bom_csv, index=False)

    # DXF output
    write_r12_dxf_polylines(polylines, out_dxf, units="m")

    console.print(f"[green]✓[/green] Route optimization complete: {ok_count}/{len(results)} route(s) found")
    console.print(f"[green]✓[/green] DXF: {out_dxf}")
    console.print(f"[green]✓[/green] Routes CSV: {out_routes_csv}")
    console.print(f"[green]✓[/green] BOM CSV: {out_bom_csv}")

    # Write a single PNG preview for all successful routes
    preview_routes = [(r.route_id, r.path) for r in results if r.status == "ok" and r.path]
    if preview_routes:
        out_png = "route_preview.png"
        if plot_routes_preview is not None:
            plot_routes_preview(layout=layout, routes=preview_routes, out_png=out_png, show_steps=False)
        console.print(f"[green]✓[/green] Preview PNG: {out_png}")

@app.command("security-coverage")
def security_coverage(
    layout_json: str = typer.Argument(..., help="Path to layout.json describing the site grid + obstacles."),
    sensors_csv: str = typer.Argument(..., help="Path to sensors.csv (sensor_id,x,y,range_cells,fov_deg,heading_deg)."),
    out_png: str = typer.Option("coverage.png", "--out-png", help="Output PNG coverage map path."),
    out_summary_json: str = typer.Option("coverage_summary.json", "--out-summary-json", help="Output summary JSON path."),
    out_blind_csv: str = typer.Option("blind_spots.csv", "--out-blind-csv", help="Output blind spots CSV path."),
):
    """
    Compute grid-based security coverage with field-of-view and obstacle occlusion.

    Inputs: layout.json + sensors.csv
    Outputs: coverage.png + coverage_summary.json + blind_spots.csv
    """
    layout = load_layout(layout_json)
    sensors = load_sensors_csv(sensors_csv)

    analyzer = CoverageAnalyzer(layout, sensors)
    cov = analyzer.coverage_grid()
    summary = analyzer.summary(cov)
    blind = analyzer.blind_spots(cov)

    # PNG visualization
    plot_coverage(layout, cov, sensors, out_png=out_png, show_blind_spots=True)

    # Summary JSON
    payload = {
        "width": summary.width,
        "height": summary.height,
        "sensors": summary.sensors,
        "coverage_pct": round(summary.coverage_pct, 3),
        "covered_cells": summary.covered_cells,
        "uncovered_cells": summary.uncovered_cells,
        "single_covered_cells": summary.single_covered_cells,
        "notes": summary.notes,
    }
    Path(out_summary_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Blind spots CSV
    import csv as _csv
    with open(out_blind_csv, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["x", "y"])
        for (x, y) in blind:
            w.writerow([x, y])

    console.print(f"[green]✓[/green] Coverage PNG: {out_png}")
    console.print(f"[green]✓[/green] Summary JSON: {out_summary_json}")
    console.print(f"[green]✓[/green] Blind spots CSV: {out_blind_csv}")
    console.print(
        f"[bold]Coverage:[/bold] {summary.coverage_pct:.2f}%  |  "
        f"Uncovered: {summary.uncovered_cells}  |  Single-covered: {summary.single_covered_cells}"
    )


if __name__ == "__main__":

    # Backwards-compatible shim:
    # Historically, this CLI supported `python src/odat2/cli.py <csv_file> --out-json ...`
    # After adding additional commands, Typer requires an explicit command name.
    # If the first non-option argument looks like a CSV path, treat it as `main`.
    import sys as _sys
    _argv = list(_sys.argv)
    _known_cmds = {"main", "route-optimize", "security-coverage"}
    _first = None
    for a in _argv[1:]:
        if a.startswith("-"):
            continue
        _first = a
        break
    if _first and (_first.lower().endswith(".csv") or _first.lower().endswith(".tsv")) and _first not in _known_cmds:
        _sys.argv = [_argv[0], "main"] + _argv[1:]

    app()
