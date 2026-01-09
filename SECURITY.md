## Reporting
This is a student project. If you discover a security issue, please open a GitHub Issue with steps to reproduce.

## Operational Security Notes (OT-friendly)
- This tool is intended to run **offline** on exported CSVs.
- The CLI performs **no network calls** and has no telemetry.
- Input CSVs may contain sensitive infrastructure identifiers (device tags, drawing IDs).
- For public demos, use sanitized or synthetic data (see `sample.csv` / `sample_clean.csv`).
- Keep dependencies minimal and patched (see `docs/pip-audit.txt` and Dependabot configuration).
