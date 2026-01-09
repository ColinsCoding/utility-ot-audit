from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt


# -----------------------------
# Determinism helpers
# -----------------------------
def set_determinism(seed: int = 7) -> None:
    # numpy RNG is controlled inside the generator; set global as well
    np.random.seed(seed)
    # help some libs be deterministic-ish (won't harm if unused)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# -----------------------------
# Synthetic OT telecom dataset
# -----------------------------
def generate_ot_telecom_dataset(n: int = 20000, seed: int = 7) -> pd.DataFrame:
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

    df = pd.DataFrame({
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
    return df


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
    # one-hot columns will appear too, leave them as-is
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

  // Optional: show feedback loop
  F -> C [style=dashed, label="tune pruning params", fontcolor="#555555"];
}
""".strip() + "\n"
    (out_dir / "workflow.dot").write_text(dot, encoding="utf-8")


# -----------------------------
# Main pipeline
# -----------------------------
def main():
    seed = 7
    n_rows = 20000

    set_determinism(seed)

    out_dir = Path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Data
    df = generate_ot_telecom_dataset(n=n_rows, seed=seed)
    df.to_csv(out_dir / "synthetic_ot_telecom.csv", index=False)

    # 2) Features/labels
    y = df["needs_engineer_review"]
    X = pd.get_dummies(df.drop(columns=["needs_engineer_review"]), columns=["device_type"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # 3) Decision Tree (regularized to avoid overfit)
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=200,
        min_samples_split=400,
        random_state=42
    )
    tree.fit(X_train, y_train)

    pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)[:, 1]

    tree_report = classification_report(y_test, pred, digits=3)
    tree_auc = float(roc_auc_score(y_test, proba))

    # 4) Rules (raw + engineering rewrite)
    raw_rules = export_text(tree, feature_names=list(X.columns))
    eng_rules = rewrite_export_text(raw_rules)

    (out_dir / "rules_raw.txt").write_text(raw_rules, encoding="utf-8")
    (out_dir / "rules_engineering.txt").write_text(eng_rules, encoding="utf-8")

    # 5) Tree DOT (model graph)
    export_graphviz(
        tree,
        out_file=str(out_dir / "tree.dot"),
        feature_names=list(X.columns),
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        special_characters=True
    )

    # 6) Tree image (matplotlib) - no Docker/Graphviz required
    plt.figure(figsize=(18, 9))
    plot_tree(
        tree,
        feature_names=list(X.columns),
        class_names=["no", "yes"],
        filled=True,
        rounded=True,
        max_depth=4,
        fontsize=8
    )
    plt.tight_layout()
    plt.savefig(out_dir / "tree_matplotlib.png", dpi=200)
    plt.close()

    # 7) Random Forest (stability)
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

    rf_report = classification_report(y_test, rf_pred, digits=3)
    rf_auc = float(roc_auc_score(y_test, rf_proba))

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    importances.to_csv(out_dir / "feature_importance.csv", header=["importance"])

    # 8) Workflow DOT (pipeline diagram)
    write_workflow_dot(out_dir)

    # 9) Manifest (reproducibility)
    manifest = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
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
            "dataset": "out/synthetic_ot_telecom.csv",
            "tree_dot": "out/tree.dot",
            "tree_png": "out/tree_matplotlib.png",
            "rules_raw": "out/rules_raw.txt",
            "rules_engineering": "out/rules_engineering.txt",
            "feature_importance": "out/feature_importance.csv",
            "workflow_dot": "out/workflow.dot",
        }
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 10) Human-readable report
    report = []
    report.append("# OT Telecom Review Aid – Synthetic Demo\n")
    report.append("## Decision Tree\n")
    report.append(f"- ROC AUC: {tree_auc:.3f}\n")
    report.append("```\n" + tree_report + "```\n")
    report.append("## Random Forest\n")
    report.append(f"- ROC AUC: {rf_auc:.3f}\n")
    report.append("```\n" + rf_report + "```\n")
    report.append("## Top Features (Forest)\n")
    report.append(importances.head(10).to_string() + "\n")
    report.append("\n## Engineering Rules (Tree)\n")
    report.append("```\n" + eng_rules + "\n```\n")

    (out_dir / "report.md").write_text("".join(report), encoding="utf-8")

    print("DONE. Wrote outputs to ./out")
    print("Open these:")
    print(" - out/tree_matplotlib.png")
    print(" - out/report.md")
    print(" - out/rules_engineering.txt")
    print(" - out/workflow.dot (pipeline diagram)")


if __name__ == "__main__":
    main()
