from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Optional


@dataclass(frozen=True)
class DocConfidenceResult:
    confidence: float          # 0..1
    days_since_verified: Optional[int]
    status: str                # "OK" | "WARN" | "ERROR" | "UNKNOWN"


def _parse_yyyy_mm_dd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def compute_doc_confidence(
    last_verified_date: Optional[str],
    *,
    today: date,
    half_life_days: float,
) -> DocConfidenceResult:
    d0 = _parse_yyyy_mm_dd(last_verified_date or "")
    if d0 is None:
        return DocConfidenceResult(confidence=0.0, days_since_verified=None, status="UNKNOWN")

    days = (today - d0).days
    if days < 0:
        # Future date is suspicious: treat as unknown rather than crashing.
        return DocConfidenceResult(confidence=0.0, days_since_verified=days, status="UNKNOWN")

    lam = math.log(2.0) / float(half_life_days)
    c = math.exp(-lam * float(days))
    c = max(0.0, min(1.0, c))

    # Deterministic thresholds (tune later)
    if c < 0.25:
        status = "ERROR"
    elif c < 0.50:
        status = "WARN"
    else:
        status = "OK"

    return DocConfidenceResult(confidence=c, days_since_verified=days, status=status)
