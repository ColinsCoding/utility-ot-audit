from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from odat2.validators.review_priority import compute_review_priority


@dataclass(frozen=True)
class ReviewPriorityUncertainty:
    mean_score: float
    std_score: float
    q05: float
    q50: float
    q95: float
    p_low: float
    p_medium: float
    p_high: float
    samples: int
    sigma_doc_confidence: float
    base_doc_confidence: float
    notes: Dict[str, Any]


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def monte_carlo_review_priority(
    *,
    errors: int,
    warnings: int,
    issue_type_counts: Dict[str, int],
    doc_confidence: float,
    sigma_doc_confidence: float = 0.05,
    n: int = 5000,
    seed: Optional[int] = 0,
) -> ReviewPriorityUncertainty:
    """
    Propagate uncertainty in doc_confidence into Review Priority Score (RPS)
    using Monte Carlo (robust near thresholds/clamps).
    """
    rng = np.random.default_rng(seed)

    C0 = float(doc_confidence)
    sig = float(sigma_doc_confidence)

    # Sample doc confidence uncertainty (assume approx Normal, clipped to [0,1])
    C = _clip01(rng.normal(loc=C0, scale=sig, size=int(n)))

    scores = np.empty_like(C, dtype=float)
    levels = np.empty(C.shape[0], dtype="U6")  # "LOW"/"MEDIUM"/"HIGH"

    for i, ci in enumerate(C):
        r = compute_review_priority(
            errors=int(errors),
            warnings=int(warnings),
            issue_type_counts=issue_type_counts,
            doc_confidence=float(ci),
        )
        scores[i] = float(r.score_0_100)
        levels[i] = r.level

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0
    q05, q50, q95 = (float(x) for x in np.quantile(scores, [0.05, 0.50, 0.95]))

    p_low = float(np.mean(levels == "LOW"))
    p_medium = float(np.mean(levels == "MEDIUM"))
    p_high = float(np.mean(levels == "HIGH"))

    # Quick analytic slope (only valid away from clamp/threshold effects):
    # score = 100 * ( ... + wS*(1-C) + ... ), wS=0.15 => dscore/dC = -15
    approx_sigma = 15.0 * sig

    return ReviewPriorityUncertainty(
        mean_score=round(mean, 2),
        std_score=round(std, 2),
        q05=round(q05, 2),
        q50=round(q50, 2),
        q95=round(q95, 2),
        p_low=round(p_low, 4),
        p_medium=round(p_medium, 4),
        p_high=round(p_high, 4),
        samples=int(scores.size),
        sigma_doc_confidence=sig,
        base_doc_confidence=round(C0, 6),
        notes={
            "approx_sigma_if_smooth": round(approx_sigma, 4),
            "warning": "Monte Carlo is recommended near score thresholds (40/70) and clamping.",
        },
    )
