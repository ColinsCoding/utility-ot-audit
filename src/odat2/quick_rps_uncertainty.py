import json
import argparse
import numpy as np

from odat2.validators.review_priority import compute_review_priority


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report_json", help="Path to JSON report produced by odat2")
    ap.add_argument("--sigma", type=float, default=0.05, help="Std dev for doc_confidence uncertainty")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rep = json.loads(open(args.report_json, "r", encoding="utf-8").read())

    drivers = rep.get("review_priority_drivers", {})
    C0 = float(drivers.get("doc_confidence", rep.get("doc_confidence", 0.0)))

    errors = int(drivers.get("errors", rep.get("errors", 0)))
    warnings = int(drivers.get("warnings", rep.get("warnings", 0)))

    issue_type_counts = {
        "orphaned_device": int(drivers.get("orphaned_device", 0)),
        "single_point_of_failure": int(drivers.get("single_point_of_failure", 0)),
    }

    rng = np.random.default_rng(args.seed)
    C = clip01(rng.normal(C0, args.sigma, size=args.n))

    scores = np.empty(args.n, dtype=float)
    levels = np.empty(args.n, dtype="U6")

    for i, ci in enumerate(C):
        r = compute_review_priority(
            errors=errors,
            warnings=warnings,
            issue_type_counts=issue_type_counts,
            doc_confidence=float(ci),
        )
        scores[i] = r.score_0_100
        levels[i] = r.level

    out = {
        "base_doc_confidence": round(C0, 6),
        "sigma_doc_confidence": args.sigma,
        "n": args.n,
        "mean_score": round(float(np.mean(scores)), 2),
        "std_score": round(float(np.std(scores, ddof=1)), 2),
        "q05": round(float(np.quantile(scores, 0.05)), 2),
        "q50": round(float(np.quantile(scores, 0.50)), 2),
        "q95": round(float(np.quantile(scores, 0.95)), 2),
        "p_low": round(float(np.mean(levels == "LOW")), 4),
        "p_medium": round(float(np.mean(levels == "MEDIUM")), 4),
        "p_high": round(float(np.mean(levels == "HIGH")), 4),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
