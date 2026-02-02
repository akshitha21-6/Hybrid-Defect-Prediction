import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ---------- PATH SETUP ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ML_DIR = ROOT / "ml"
sys.path.append(str(ML_DIR))
# -------------------------------

from preprocess import preprocess_data
from feature_selection import (
    fs_none, fs_rfe_rf, fs_ga, fs_aco, fs_pso, fs_hybrid
)
from models import train_model
from evaluation import plot_confusion, plot_roc

st.set_page_config(
    page_title="Hybrid Software Defect Prediction",
    layout="wide"
)

st.title("🧠 Hybrid Swarm-Optimized Software Defect Prediction System")

# -------- AUTO-DETECT DATASETS --------
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    st.error("❌ No CSV datasets found inside data/ folder")
    st.stop()

DATASETS = {f.stem.upper(): f for f in csv_files}
# ------------------------------------

FS_METHODS = {
    "None": fs_none,
    "RFE-RF": fs_rfe_rf,
    "GA": fs_ga,
    "ACO": fs_aco,
    "PSO": fs_pso,
    "Hybrid": fs_hybrid,
}

st.sidebar.header("⚙️ Configuration")
dataset = st.sidebar.selectbox("📂 Dataset", list(DATASETS.keys()))
fs_choice = st.sidebar.selectbox("🧠 Feature Selection", list(FS_METHODS.keys()))
run = st.sidebar.button("🚀 Run Prediction")

if run:
    dataset_path = DATASETS[dataset]

    X, y, names = preprocess_data(dataset_path)
    X_sel, selected_features = FS_METHODS[fs_choice](X, y, names)

    results = train_model(X_sel, y)

    st.subheader("📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", round(results["accuracy"], 3))
    c2.metric("Precision", round(results["precision"], 3))
    c3.metric("Recall", round(results["recall"], 3))
    c4.metric("F1 Score", round(results["f1"], 3))

    st.subheader("🧩 Selected Features")
    st.write(selected_features)

    st.subheader("📉 Confusion Matrix")
    st.pyplot(plot_confusion(results["y_test"], results["y_pred"]))

    st.subheader("📈 ROC Curve")
    roc_fig, auc_val = plot_roc(results["y_test"], results["y_prob"])
    st.pyplot(roc_fig)
    st.write(f"**AUC:** {auc_val:.3f}")

    st.subheader("📋 Comparison Table")
    st.table(pd.DataFrame([{
        "Dataset": dataset,
        "Feature Selection": fs_choice,
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1": results["f1"],
        "AUC": auc_val
    }]))

    st.subheader("📄 Final Auto-Generated Report")
    st.write(
        f"""
        The **{dataset} dataset** was analyzed using a **Random Forest classifier**
        with **{fs_choice} feature selection** and **SMOTE** for imbalance handling.

        The hybrid swarm-based approach improves defect prediction accuracy
        while reducing the number of required software metrics.
        """
    )