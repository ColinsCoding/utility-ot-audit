# ODAT2 — Operations Data Audit Tool

ODAT2 is a small, **deterministic** command-line tool that validates telecommunications / security cable schedule data exported from engineering drawings (CSV). It produces an auditable issue list (terminal summary + JSON/HTML reports) to reduce rework and improve drawing/data consistency.

This project is intentionally **not** a flashy ML demo: it focuses on repeatability, safety, and engineering judgment.

## What it checks (today)

- **Cable sanity**: flags placeholder / suspicious cable lengths.
- **Device tag format**: flags nonstandard device identifiers.
- **Topology integrity**: finds orphaned devices and single points of failure in the network graph.

## Why this is relevant to utility telecom / OT work

In electric utility environments (substations + office buildings), telecom/security documentation quality matters. ODAT2 models a realistic workflow:

1. Export a CSV from drawing data (or a cable schedule).
2. Run deterministic checks (no network calls; local-only).
3. Review a summary table and a structured report.

This mirrors the kind of QA/validation engineers do alongside AutoCAD/Visio drawing management.

## Quickstart (Windows 11)

From the repo root:

```powershell
py -m venv .venv
\.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .

# Run an audit
odat2 .\sample.csv --out-json odat_report.json

# Run tests
python -m pip install -e .[dev]
python -m pytest -q
```

If you prefer running without an editable install, you can also set `PYTHONPATH`:

```powershell
$env:PYTHONPATH=".\src"
\.\.venv\Scripts\python.exe .\src\odat2\cli.py .\sample.csv --out-json odat_report.json
```

## Output

- **Terminal**: a safe summary table (counts by severity/type).
- **JSON**: structured issues for review or downstream tooling.
- **HTML**: a simple report view (optional).

## Repository layout

```
src/odat2/          # core package
  io/               # CSV ingest
  validators/       # deterministic checks
  reports/          # JSON/HTML/terminal output
tests/              # pytest tests
sample.csv          # sanitized example input
```

## Security and data handling

See `SECURITY.md`. Short version: **do not** upload proprietary drawings/exports, and prefer summary outputs in sensitive environments.

## Roadmap (small + practical)

- `--counts-only` mode: print summary tables without listing sample issues.
- Chunked CSV ingest for very large exports.
- Optional C acceleration for hot paths (counting / scanning) with identical results.

## Quick Demo

### FAIL example (risk identified)
```powershell
python src/odat2/cli.py sample.csv --today 2026-01-08

## Security
- Offline CLI (no network calls / no telemetry).
- Dependency checks: see `docs/pip-audit.txt`. Dependabot is enabled for weekly updates.
