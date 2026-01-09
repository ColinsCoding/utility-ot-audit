# Models

This project favors deterministic, auditable models over opaque ML. The goal is to produce repeatable, explainable signals that help engineers prioritize review work.

## Documentation Confidence Decay (Half-Life Model)

We model the confidence that documentation (or exported design data) is current as an exponential decay function of time since the last verification date.

### Inputs
- `last_verified_date` (CSV field): `YYYY-MM-DD`
- `today` (CLI override for reproducibility): `YYYY-MM-DD`
- `half_life_days` (CLI option): positive number of days

Let:
- `t` = `days_since_verified` = `(today - last_verified_date).days`
- `T½` = `half_life_days`

### Model
We compute a decay rate:

\[
\lambda = \ln(2) / T_{1/2}
\]

and confidence:

\[
C(t) = e^{-\lambda t}
\]

This is equivalent to the half-life form:

\[
C(t) = 2^{-t/T_{1/2}}
\]

Properties:
- If `t = 0`, then `C = 1.0` (freshly verified).
- If `t = T½`, then `C = 0.5` (confidence halves).
- As `t` grows, `C(t)` monotonically decreases toward 0.

### Safety / Determinism
- If `last_verified_date` is missing or cannot be parsed, we return `status = "UNKNOWN"` and `confidence = 0.0`.
- If `last_verified_date` is in the future (`t < 0`), we return `status = "UNKNOWN"` and `confidence = 0.0`.
- Confidence is clamped to `[0.0, 1.0]` for safety.

### Status Thresholds
We map confidence to a deterministic status:

- `C < 0.25` → `ERROR`
- `0.25 ≤ C < 0.50` → `WARN`
- `C ≥ 0.50` → `OK`
- parse failure or future date → `UNKNOWN`

These thresholds are intentionally simple and can be tuned later.

### Rationale
Utilities and OT environments often require decisions that are:
- explainable to engineers and reviewers,
- reproducible across machines and time,
- auditable after the fact.

Exponential half-life decay provides a mathematically grounded, transparent way to prioritize stale documentation without requiring large labeled datasets.
