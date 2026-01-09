# Security & Data Handling

ODAT2 is designed to be useful in operational technology (OT) / utility environments, where data is often sensitive.

## Rules of thumb

1. **Do not commit proprietary data**
   - Do not add real utility drawings, substation diagrams, internal Visio/AutoCAD files, or raw exports to this repo.
   - Keep only sanitized, synthetic, or publicly shareable example inputs (like `sample.csv`).

2. **Treat reports as sensitive**
   - JSON/HTML reports may contain device tags, cable IDs, or drawing IDs.
   - Store reports locally and share only if approved.

3. **Prefer summary output in sensitive environments**
   - When working with real data, prefer printing counts/tables over listing individual records.
   - (Planned) `--counts-only` mode will support this explicitly.

4. **Local-only by design**
   - ODAT2 makes no network calls and sends no telemetry.
   - Keep it that way unless you have explicit authorization.

## Safe ways to practice

- Generate synthetic CSVs with fake device tags and cable IDs.
- Mask identifiers (e.g., hash or replace device tags) before sharing examples.
- Document assumptions and limits rather than copying logs or internal details.

## Reporting issues

If you find a security-relevant bug (e.g., data leakage in output), open a GitHub issue describing the problem **without** uploading sensitive data.
