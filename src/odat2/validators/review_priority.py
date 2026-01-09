from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, Tuple


def _lognorm(x: int, k: int) -> float:
    """Log-normalize count to [0,1] with saturation around k."""
    if x <= 0:
        return 0.0
    if k <= 0:
        return 1.0
    v = math.log1p(float(x)) / math.log1p(float(k))
    return max(0.0, min(1.0, v))


@dataclass(frozen=True)
class ReviewPriorityResult:
    score_0_100: float
    level: str  # "LOW" | "MEDIUM" | "HIGH"
    drivers: Dict[str, Any]


def compute_review_priority(
    *,
    errors: int,
    warnings: int,
    issue_type_counts: Dict[str, int],
    doc_confidence: float,
) -> ReviewPriorityResult:
    # Components in [0,1]
    E = _lognorm(errors, k=5)
    W = _lognorm(warnings, k=10)

    orphaned = int(issue_type_counts.get("orphaned_device", 0))
    spof = int(issue_type_counts.get("single_point_of_failure", 0))

    O = _lognorm(orphaned, k=3)
    P = 1.0 if spof > 0 else 0.0

    # Staleness risk in [0,1]
    C = max(0.0, min(1.0, float(doc_confidence)))
    S = 1.0 - C

    # Weights sum to 1 for interpretability (convex combination)
    wE, wO, wP, wS, wW = 0.40, 0.20, 0.15, 0.15, 0.10
    r = wE * E + wO * O + wP * P + wS * S + wW * W
    score = 100.0 * max(0.0, min(1.0, r))

    if score >= 70.0:
        level = "HIGH"
    elif score >= 40.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    drivers = {
        "errors": errors,
        "warnings": warnings,
        "orphaned_device": orphaned,
        "single_point_of_failure": spof,
        "doc_confidence": round(C, 6),
        "doc_staleness": round(S, 6),
        "components_0_1": {"E": round(E, 4), "O": round(O, 4), "P": round(P, 4), "S": round(S, 4), "W": round(W, 4)},
        "weights": {"E": wE, "O": wO, "P": wP, "S": wS, "W": wW},
    }

    return ReviewPriorityResult(score_0_100=round(score, 2), level=level, drivers=drivers)
