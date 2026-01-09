# ot_tree_pipeline.py
from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt

import csv
from typing import List, Dict, Any

from typing import Tuple

# -----------------------------
# Determinism helpers
# -----------------------------
def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------
# Synthetic OT telecom dataset
# -----------------------------
def generate_ot_telecom_dataset(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    device_type = rng.choice(
        ["relay", "switch", "radio", "rtu"],
        size=n,
        p=[0.45, 0.25, 0.20, 0.10]
    )

    # latent "site quality" induces correlations
    site_quality = rng.normal(0, 1, size=n)

    doc_age_days = np.clip(rng.gamma(shape=2.0, scale=220.0, size=n) - 120 * site_quality, 0, 2000)
    has_drawing_id = (rng.random(n) < sigmoid(1.2 - 0.002 * doc_age_days + 0.8 * site_quality)).astype(int)

    # physical-ish comms
    fiber_length_m = np.clip(rng.gamma(shape=2.5, scale=120.0, size=n) + 60 * (device_type == "radio"), 5, 1500)
    snr_db = np.clip(rng.normal(25 + 3 * site_quality - 0.004 * fiber_length_m, 4.0, size=n), 0, 40)
    packet_loss_pct = np.clip(rng.normal(0.4 - 0.12 * site_quality + 0.03 * (snr_db < 12), 0.25, size=n), 0, 5)
    latency_ms = np.clip(rng.normal(6 + 4 * (device_type == "radio") + 1.5 * packet_loss_pct, 2.5, size=n), 0, 50)

    # design/topology
    path_diverse = (rng.random(n) < sigmoid(0.6 + 0.7 * site_quality - 0.3 * (device_type == "radio"))).astype(int)
    vlan_ok = (rng.random(n) < sigmoid(1.0 + 0.5 * site_quality - 0.8 * (device_type == "relay"))).astype(int)

    margin_db = np.clip(
        rng.normal(6 + 2.0 * site_quality + 0.35 * (snr_db - 20) - 1.2 * packet_loss_pct - 0.05 * latency_ms, 3.5, size=n),
        -15, 20
    )

    spof_risk_score = np.clip(
        0.55 * (1 - path_diverse) + 0.25 * (device_type == "switch") + 0.20 * (device_type == "rtu") + rng.normal(0, 0.12, size=n),
        0, 1
    )

    # probabilistic label (anti-lookup-table)
    risk = (
        1.2 * (has_drawing_id == 0)
        + 1.0 * (doc_age_days > 900)
        + 1.2 * (margin_db < 2)
        + 1.0 * (packet_loss_pct > 1.0)
        + 1.0 * (snr_db < 12)
        + 1.3 * (spof_risk_score > 0.6)
        + 0.6 * (vlan_ok == 0)
        + 0.4 * (device_type == "relay")
    )
    risk = risk + rng.normal(0, 0.9, size=n)
    p = sigmoid(-1.0 + 0.9 * risk)
    needs_engineer_review = (rng.random(n) < p).astype(int)

    return pd.DataFrame({
        "device_type": device_type,
        "doc_age_days": doc_age_days,
        "has_drawing_id": has_drawing_id,
        "fiber_length_m": fiber_length_m,
        "snr_db": snr_db,
        "packet_loss_pct": packet_loss_pct,
        "latency_ms": latency_ms,
        "path_diverse": path_diverse,
        "vlan_ok": vlan_ok,
        "margin_db": margin_db,
        "spof_risk_score": spof_risk_score,
        "needs_engineer_review": needs_engineer_review,
    })


# -----------------------------
# Engineering rewrite of rules
# -----------------------------
FEATURE_REWRITE = {
    "doc_age_days": "Documentation age (days since last verified)",
    "has_drawing_id": "Drawing reference present (AutoCAD/Visio sheet ID)",
    "margin_db": "EMC margin to limit line (dB)",
    "spof_risk_score": "Single-point-of-failure risk score (0..1)",
    "packet_loss_pct": "Packet loss (%)",
    "snr_db": "Signal-to-noise ratio (dB)",
    "latency_ms": "Latency (ms)",
    "path_diverse": "A/B path diversity present",
    "vlan_ok": "VLAN assignment matches design",
}


def rewrite_rule_line(line: str) -> str:
    m = re.search(r"([a-zA-Z0-9_]+)\s*(<=|>=|<|>)\s*([0-9.]+)", line)
    if not m:
        return line.rstrip()

    feat, op, val = m.group(1), m.group(2), float(m.group(3))
    eng = FEATURE_REWRITE.get(feat, feat)
    prefix = line.split(feat)[0]

    if feat == "margin_db" and op in ("<=", "<") and val <= 3.0:
        return f"{prefix}{eng} {op} {val:.2f} → Low margin. Re-run test with tighter span / more averaging; confirm setup repeatability."
    if feat == "doc_age_days" and op in (">=", ">") and val >= 900:
        return f"{prefix}{eng} {op} {val:.0f} → Docs stale. Verify as-builts, port maps, BOM; update drawing package."
    if feat == "spof_risk_score" and op in (">=", ">") and val >= 0.6:
        return f"{prefix}{eng} {op} {val:.2f} → SPOF risk. Verify redundant comms paths, conduit diversity, ring design."

    return f"{prefix}{eng} {op} {val}"


def rewrite_export_text(tree_text: str) -> str:
    return "\n".join(rewrite_rule_line(line) for line in tree_text.splitlines())


# -----------------------------
# Workflow (deterministic) flow diagram
# -----------------------------
def write_workflow_dot(out_dir: Path) -> None:
    dot = r"""
digraph OT_AUDIT_PIPELINE {
  rankdir=LR;
  node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Arial"];

  A [label="Generate synthetic OT telecom dataset\n(df: devices/links/features)"];
  B [label="Split train/test\n(stratified)"];
  C [label="Train Decision Tree\n(max_depth, min_samples_leaf)"];
  D [label="Evaluate Tree\n(classification report, AUC)"];
  E [label="Export Tree artifacts\n(tree.dot, tree_matplotlib.png)"];
  F [label="Export Rules\n(raw + engineering rewrite)"];
  G [label="Train Random Forest\n(stability + importances)"];
  H [label="Export Forest artifacts\n(feature_importance.csv)"];
  I [label="Write manifest + outputs\n(reproducible run)"];

  A -> B -> C -> D -> E -> F -> I;
  D -> G -> H -> I;

  F -> C [style=dashed, label="tune pruning params", fontcolor="#555555"];
}
""".strip() + "\n"
    (out_dir / "workflow.dot").write_text(dot, encoding="utf-8")


# -----------------------------
# CLI + Main pipeline
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ot_tree_pipeline", description="Deterministic OT telecom tree/forest demo pipeline.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for determinism (default: 7)")
    p.add_argument("--rows", type=int, default=20000, help="Number of synthetic rows to generate (default: 20000)")
    p.add_argument("--out", type=str, default="runs", help="Base output directory (default: runs)")
    return p.parse_args()

SEVERITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "INFO": 3}

def constraints_for_row(row: pd.Series) -> List[Dict[str, Any]]:
    """
    Deterministic constraint checks using ONLY columns that exist in your synthetic dataset.
    Returns a list of findings dicts.
    """
    findings: List[Dict[str, Any]] = []

    if int(row["has_drawing_id"]) == 0:
        findings.append({
            "severity": "P0",
            "finding_code": "MISSING_DRAWING_REF",
            "engineering_explanation": "No drawing reference present (AutoCAD/Visio sheet ID missing).",
            "recommended_action": "Add drawing_id / sheet reference; update drawing package before signoff.",
            "source": "constraint",
        })

    if float(row["spof_risk_score"]) > 0.6 and int(row["path_diverse"]) == 0:
        findings.append({
            "severity": "P0",
            "finding_code": "SPOF_NO_PATH_DIVERSITY",
            "engineering_explanation": "High SPOF risk and no A/B path diversity indicated.",
            "recommended_action": "Verify redundant comms paths, conduit diversity, ring design; document A/B paths.",
            "source": "constraint",
        })

    if int(row["vlan_ok"]) == 0:
        findings.append({
            "severity": "P1",
            "finding_code": "VLAN_MISMATCH",
            "engineering_explanation": "VLAN assignment does not match design intent (vlan_ok=0).",
            "recommended_action": "Validate port maps and switch configuration; update drawings if design changed.",
            "source": "constraint",
        })

    if float(row["margin_db"]) < 3.0:
        findings.append({
            "severity": "P1",
            "finding_code": "LOW_EMC_MARGIN",
            "engineering_explanation": f"Low margin to limit line (margin_db={float(row['margin_db']):.2f} dB).",
            "recommended_action": "Re-run test with tighter span / increased averaging; confirm setup repeatability.",
            "source": "constraint",
        })

    if float(row["doc_age_days"]) > 900:
        findings.append({
            "severity": "P2",
            "finding_code": "DOC_STALE",
            "engineering_explanation": f"Documentation appears stale (doc_age_days={float(row['doc_age_days']):.0f}).",
            "recommended_action": "Verify as-builts, port maps, BOM; refresh drawing package.",
            "source": "constraint",
        })

    return findings


def write_findings_csv(out_dir: Path, run_id: str, df_test_raw: pd.DataFrame, prob_review: np.ndarray, max_assets: int = 200) -> None:
    """
    Writes findings.csv: one row per finding per asset.
    Deterministic ordering: highest model score first, then by asset_id.
    """
    findings_path = out_dir / "findings.csv"

    # Deterministic ordering
    order = np.lexsort((df_test_raw.index.to_numpy(), -prob_review))
    selected = order[: min(max_assets, len(order))]

    with findings_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "asset_id", "device_type",
            "severity", "finding_code",
            "engineering_explanation", "recommended_action",
            "source", "model_score"
        ])

        for pos in selected:
            asset_id = int(df_test_raw.index[pos])
            row = df_test_raw.loc[asset_id]
            score = float(prob_review[pos])

            findings = constraints_for_row(row)

            # If no constraints hit, still include a model triage row
            if not findings:
                findings.append({
                    "severity": "INFO",
                    "finding_code": "NO_CONSTRAINT_FLAGS",
                    "engineering_explanation": "No deterministic constraint violations detected.",
                    "recommended_action": "Continue; normal spot-checks apply.",
                    "source": "constraint",
                })

            # Always include a model triage suggestion row (engineer-readable, not “ML says so”)
            if score >= 0.80:
                model_row = ("P1", "MODEL_HIGH_PRIORITY", "High priority for engineering review based on combined indicators.", "Route to engineer queue; verify docs + redundancy + retest repeatability.")
            elif score >= 0.50:
                model_row = ("P2", "MODEL_MED_PRIORITY", "Medium priority for review based on combined indicators.", "Check docs and key risk indicators; retest if borderline.")
            else:
                model_row = ("INFO", "MODEL_LOW_PRIORITY", "Low priority for review based on combined indicators.", "Continue; normal checks apply.")

            findings.append({
                "severity": model_row[0],
                "finding_code": model_row[1],
                "engineering_explanation": f"{model_row[2]} (score={score:.2f})",
                "recommended_action": model_row[3],
                "source": "decision_tree",
            })

            findings.sort(key=lambda d: (SEVERITY_RANK.get(d["severity"], 99), d["finding_code"]))

            for fd in findings:
                w.writerow([
                    run_id, asset_id, str(row["device_type"]),
                    fd["severity"], fd["finding_code"],
                    fd["engineering_explanation"], fd["recommended_action"],
                    fd["source"], f"{score:.4f}"
                ])

    print(f"Wrote {findings_path}")

def write_findings_summary(out_dir: Path) -> None:
    """
    Reads findings.csv and writes findings_summary.csv (counts by severity and finding_code).
    Must run AFTER findings.csv is written.
    """
    findings_path = out_dir / "findings.csv"
    summary_path = out_dir / "findings_summary.csv"

    if not findings_path.exists():
        raise FileNotFoundError(f"{findings_path} does not exist yet")

    df = pd.read_csv(findings_path)

    sev_counts = df.groupby("severity").size().reset_index(name="count")
    sev_counts["summary_type"] = "by_severity"
    sev_counts = sev_counts.rename(columns={"severity": "key"})

    code_counts = df.groupby("finding_code").size().reset_index(name="count").sort_values("count", ascending=False)
    code_counts["summary_type"] = "by_finding_code"
    code_counts = code_counts.rename(columns={"finding_code": "key"})

    summary = pd.concat([sev_counts, code_counts], ignore_index=True)
    summary = summary[["summary_type", "key", "count"]]

    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

RISK_FEATURES = [
    "doc_age_days",
    "has_drawing_id",
    "margin_db",
    "snr_db",
    "packet_loss_pct",
    "latency_ms",
    "spof_risk_score",
    "path_diverse",
    "vlan_ok",
]

# For each feature: True means "higher is worse", False means "lower is worse"
# We'll transform everything so that "higher = worse" in risk-space.
HIGH_IS_WORSE = {
    "doc_age_days": True,
    "has_drawing_id": False,   # 0 missing is worse -> invert
    "margin_db": False,        # low margin is worse -> invert
    "snr_db": False,           # low SNR is worse -> invert
    "packet_loss_pct": True,
    "latency_ms": True,
    "spof_risk_score": True,
    "path_diverse": False,     # 0 (no diversity) is worse -> invert
    "vlan_ok": False,          # 0 mismatch is worse -> invert
}

def fit_minmax_risk_scaler(df_train_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Fit min/max on train only (prevents leakage).
    Returns (mins, maxs) for RISK_FEATURES.
    """
    mins = df_train_raw[RISK_FEATURES].min()
    maxs = df_train_raw[RISK_FEATURES].max()
    # avoid divide-by-zero if a column is constant
    maxs = maxs.where((maxs - mins) != 0, mins + 1.0)
    return mins, maxs

def transform_to_risk_space(df_raw: pd.DataFrame, mins: pd.Series, maxs: pd.Series) -> np.ndarray:
    """
    Min-max scale each feature to [0,1] using train mins/maxs,
    then orient so that higher = worse for every dimension.
    Returns a numpy array shape (n, d).
    """
    X = df_raw[RISK_FEATURES].copy()

    # min-max to [0,1]
    X = (X - mins) / (maxs - mins)
    X = X.clip(0.0, 1.0)

    # orient: higher should mean worse
    invert_cols = [c for c in RISK_FEATURES if not HIGH_IS_WORSE[c]]
    X[invert_cols] = 1.0 - X[invert_cols]


    return X.to_numpy(dtype=float)

def compute_distance_to_worst(risk_X: np.ndarray) -> np.ndarray:
    """
    Euclidean distance to the 'worst corner' (all ones) in risk-space.
    Smaller = closer to worst-case.
    """
    worst = np.ones((risk_X.shape[1],), dtype=float)
    diff = risk_X - worst
    return np.sqrt(np.sum(diff * diff, axis=1))

def compute_cosine_similarity_to_centroid(risk_X: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between each row vector and centroid.
    Returns values in [-1, 1] (usually [0,1] here because risk features are nonnegative).
    """
    eps = 1e-12
    x_norm = np.linalg.norm(risk_X, axis=1) + eps
    c_norm = np.linalg.norm(centroid) + eps
    dots = risk_X @ centroid
    return dots / (x_norm * c_norm)

def fit_review_centroid(risk_X_train: np.ndarray, y_train: pd.Series) -> np.ndarray:
    """
    Centroid of risk-space vectors where y_train==1 (review-needed).
    If no positives exist (unlikely), falls back to overall mean.
    """
    mask = (y_train.to_numpy() == 1)
    if mask.any():
        return risk_X_train[mask].mean(axis=0)
    return risk_X_train.mean(axis=0)

def main():
    args = parse_args()
    seed = int(args.seed)
    n_rows = int(args.rows)

    set_determinism(seed)

    # run folder: runs/run_YYYYMMDD_HHMMSS
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_out = Path(args.out)
    out_dir = base_out / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = generate_ot_telecom_dataset(args.rows, args.seed)

    # Keep raw columns (we'll add engineered features here)
    y = df["needs_engineer_review"]
    df_raw = df.drop(columns=["needs_engineer_review"]).copy()

    # Split indices FIRST so we can fit scalers on train only
    idx_train, idx_test = train_test_split(
        df.index,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    df_train_raw = df_raw.loc[idx_train]
    df_test_raw  = df_raw.loc[idx_test]
    y_train = y.loc[idx_train]
    y_test  = y.loc[idx_test]

    # ---- NEW: fit scaler on train only, compute risk-space vectors ----
    mins, maxs = fit_minmax_risk_scaler(df_train_raw)
    risk_train = transform_to_risk_space(df_train_raw, mins, maxs)
    risk_test  = transform_to_risk_space(df_test_raw,  mins, maxs)

    # ---- NEW: distance_to_worst for ALL rows (train+test) deterministically ----
    dist_train = compute_distance_to_worst(risk_train)
    dist_test  = compute_distance_to_worst(risk_test)

    # ---- NEW: similarity_to_review (cosine sim to centroid of y=1 in TRAIN) ----
    centroid = fit_review_centroid(risk_train, y_train)
    sim_train = compute_cosine_similarity_to_centroid(risk_train, centroid)
    sim_test  = compute_cosine_similarity_to_centroid(risk_test,  centroid)

    # Attach new engineered features back to raw frames (so they can be used by the tree)
    df_train_raw = df_train_raw.copy()
    df_test_raw = df_test_raw.copy()
    df_train_raw["distance_to_worst"] = dist_train
    df_test_raw["distance_to_worst"] = dist_test
    df_train_raw["similarity_to_review"] = sim_train
    df_test_raw["similarity_to_review"] = sim_test

    # Recombine for saving the full dataset artifact (optional but nice)
    df_out = pd.concat([df_train_raw, df_test_raw]).loc[df.index]  # same original ordering
    df_out["needs_engineer_review"] = y
    df_out.to_csv(out_dir/"synthetic_ot_telecom.csv", index=False)

    # Now build model features from raw + engineered features
    X_all = pd.get_dummies(df_out.drop(columns=["needs_engineer_review"]), columns=["device_type"], drop_first=True)
    X_train = X_all.loc[idx_train]
    X_test  = X_all.loc[idx_test]

    # IMPORTANT: use the engineered raw test rows (includes distance/similarity)
    df_test_raw = df_test_raw  # keep the earlier engineered df_test_raw



    # 3) Decision Tree
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=200,
        min_samples_split=400,
        random_state=42
    )
    tree.fit(X_train, y_train)

    pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)[:, 1]
    write_findings_csv(out_dir, run_id, df_test_raw, proba, max_assets=200)
    write_findings_summary(out_dir)

    tree_auc = float(roc_auc_score(y_test, proba))
    tree_report = classification_report(y_test, pred, digits=3)

    raw_rules = export_text(tree, feature_names=list(X_all.columns))
    eng_rules = rewrite_export_text(raw_rules)

    (out_dir / "rules_raw.txt").write_text(raw_rules, encoding="utf-8")
    (out_dir / "rules_engineering.txt").write_text(eng_rules, encoding="utf-8")

    export_graphviz(
        tree,
        out_file=str(out_dir / "tree.dot"),
        feature_names=list(X_all.columns),
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        special_characters=True
    )

    plt.figure(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=list(X_all.columns),
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_dir / "tree_matplotlib.png", dpi=200)
    plt.close()

    # 4) Random Forest
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=100,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)

    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)
    rf_auc = float(roc_auc_score(y_test, rf_proba))
    rf_report = classification_report(y_test, rf_pred, digits=3)

    importances = pd.Series(rf.feature_importances_, index=X_all.columns).sort_values(ascending=False)
    importances.to_csv(out_dir / "feature_importance.csv", header=["importance"])

    # 5) Pipeline diagram
    write_workflow_dot(out_dir)

    # 6) Manifest (reproducibility)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "seed": seed,
        "rows": n_rows,
        "split": {"test_size": 0.25, "random_state": 42, "stratify": True},
        "tree_params": tree.get_params(),
        "rf_params": rf.get_params(),
        "metrics": {
            "tree_auc": tree_auc,
            "rf_auc": rf_auc
        },
        "outputs": {
            "dataset": str((out_dir / "synthetic_ot_telecom.csv").as_posix()),
            "tree_dot": str((out_dir / "tree.dot").as_posix()),
            "tree_png": str((out_dir / "tree_matplotlib.png").as_posix()),
            "rules_raw": str((out_dir / "rules_raw.txt").as_posix()),
            "rules_engineering": str((out_dir / "rules_engineering.txt").as_posix()),
            "feature_importance": str((out_dir / "feature_importance.csv").as_posix()),
            "workflow_dot": str((out_dir / "workflow.dot").as_posix()),
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    manifest["risk_features"] = RISK_FEATURES
    manifest["risk_mins"] = mins.to_dict()
    manifest["risk_maxs"] = maxs.to_dict()
    manifest["review_centroid"] = centroid.tolist()

    # 7) Human report
    report = []
    report.append("# OT Telecom Review Aid – Synthetic Demo\n\n")
    report.append(f"- Run: `{run_id}`\n")
    report.append(f"- Seed: `{seed}`\n")
    report.append(f"- Rows: `{n_rows}`\n\n")

    report.append("## Decision Tree\n")
    report.append(f"- ROC AUC: {tree_auc:.3f}\n\n")
    report.append("```\n" + tree_report + "```\n\n")

    report.append("## Random Forest\n")
    report.append(f"- ROC AUC: {rf_auc:.3f}\n\n")
    report.append("```\n" + rf_report + "```\n\n")

    report.append("## Top Features (Forest)\n")
    report.append(importances.head(10).to_string() + "\n\n")

    report.append("## Engineering Rules (Tree)\n")
    report.append("```\n" + eng_rules + "\n```\n")

    (out_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print(f"DONE. Outputs written to: {out_dir}")
    print("Open:")
    print(f" - {out_dir / 'tree_matplotlib.png'}")
    print(f" - {out_dir / 'report.md'}")
    print(f" - {out_dir / 'rules_engineering.txt'}")
    print(f" - {out_dir / 'workflow.dot'}")


if __name__ == "__main__":
    main()
