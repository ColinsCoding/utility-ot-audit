import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import export_graphviz

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_ot_telecom_dataset(n=20000, seed=7):
    rng = np.random.default_rng(seed)

    device_type = rng.choice(["relay", "switch", "radio", "rtu"], size=n, p=[0.45, 0.25, 0.20, 0.10])

    # Latent "site quality" drives correlations (important to avoid toy data)
    site_quality = rng.normal(0, 1, size=n)  # higher = better

    doc_age_days = np.clip(rng.gamma(shape=2.0, scale=220.0, size=n) - 120*site_quality, 0, 2000)
    has_drawing_id = (rng.random(n) < sigmoid(1.2 - 0.002*doc_age_days + 0.8*site_quality)).astype(int)

    # Physical-ish comms
    fiber_length_m = np.clip(rng.gamma(shape=2.5, scale=120.0, size=n) + 60*(device_type=="radio"), 5, 1500)
    snr_db = np.clip(rng.normal(25 + 3*site_quality - 0.004*fiber_length_m, 4.0, size=n), 0, 40)
    packet_loss_pct = np.clip(rng.normal(0.4 - 0.12*site_quality + 0.03*(snr_db < 12), 0.25, size=n), 0, 5)
    latency_ms = np.clip(rng.normal(6 + 4*(device_type=="radio") + 1.5*packet_loss_pct, 2.5, size=n), 0, 50)

    # Design / topology
    path_diverse = (rng.random(n) < sigmoid(0.6 + 0.7*site_quality - 0.3*(device_type=="radio"))).astype(int)
    vlan_ok = (rng.random(n) < sigmoid(1.0 + 0.5*site_quality - 0.8*(device_type=="relay"))).astype(int)

    # Margin to limits (can be negative). Correlated with snr/packet_loss/latency.
    margin_db = np.clip(
        rng.normal(6 + 2.0*site_quality + 0.35*(snr_db-20) - 1.2*packet_loss_pct - 0.05*latency_ms, 3.5, size=n),
        -15, 20
    )

    # SPOF risk as a continuous score
    spof_risk_score = np.clip(
        0.55*(1 - path_diverse) + 0.25*(device_type=="switch") + 0.20*(device_type=="rtu") + rng.normal(0, 0.12, size=n),
        0, 1
    )

    # Risk score -> probability -> label (NO deterministic label!)
    # This is the key anti-overfit move.
    risk = (
        1.2*(has_drawing_id == 0)
        + 1.0*(doc_age_days > 900)
        + 1.2*(margin_db < 2)
        + 1.0*(packet_loss_pct > 1.0)
        + 1.0*(snr_db < 12)
        + 1.3*(spof_risk_score > 0.6)
        + 0.6*(vlan_ok == 0)
        + 0.4*(device_type == "relay")
    )
    risk = risk + rng.normal(0, 0.9, size=n)  # add noise so it isn't a lookup-table

    p = sigmoid(-1.0 + 0.9*risk)  # baseline shift + scale
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

def main():
    df = generate_ot_telecom_dataset(n=20000, seed=7)

    y = df["needs_engineer_review"]
    X = pd.get_dummies(df.drop(columns=["needs_engineer_review"]), columns=["device_type"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # SAFETY AGAINST OVERFITTING: restrict the tree hard
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=200,
        min_samples_split=400,
        random_state=42
    )
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    proba = tree.predict_proba(X_test)[:, 1]

    print("=== Decision Tree ===")
    print(classification_report(y_test, pred, digits=3))
    print("ROC AUC:", round(roc_auc_score(y_test, proba), 3))
    print("\nRULES:\n")
    print(export_text(tree, feature_names=list(X.columns)))

    export_graphviz(
    tree,
    out_file="tree.dot",
    feature_names=list(X.columns),
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
    special_characters=True
    )
    print("Wrote tree.dot")

    # Forest for stability + feature importances
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
    plt.savefig("tree_matplotlib.png", dpi=200)
    print("Wrote tree_matplotlib.png")


    print("\n=== Random Forest ===")
    print(classification_report(y_test, rf_pred, digits=3))
    print("ROC AUC:", round(roc_auc_score(y_test, rf_proba), 3))

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop features:\n", importances.head(10))

if __name__ == "__main__":
    main()
