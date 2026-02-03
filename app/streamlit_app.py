import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# ================= PATH SETUP =================
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ML_DIR = ROOT / "ml"
sys.path.append(str(ML_DIR))
# =============================================

from preprocess import preprocess_data
from feature_selection import (
    fs_none, fs_rfe_rf, fs_ga, fs_aco, fs_pso, fs_hybrid
)
from models import train_model
from evaluation import plot_confusion, plot_roc

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Hybrid Software Defect Prediction",
    layout="wide"
)

st.title("🧠 Hybrid Swarm-Optimized Software Defect Prediction System")
st.write(
    "This system predicts **defective software modules** using "
    "static swarm-based feature selection and machine learning."
)

# ================= LOAD DATASETS =================
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    st.error("❌ No CSV datasets found in data/ folder")
    st.stop()

DATASETS = {f.stem.upper(): f for f in csv_files}

# ================= FEATURE SELECTION METHODS =================
FS_METHODS = {
    "None (All Features)": fs_none,
    "RFE-RF": fs_rfe_rf,
    "Genetic Algorithm": fs_ga,
    "Ant Colony Optimization": fs_aco,
    "Particle Swarm Optimization": fs_pso,
    "Hybrid (GA + PSO)": fs_hybrid,
}

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Configuration")

dataset = st.sidebar.selectbox("📂 Dataset", list(DATASETS.keys()))
fs_choice = st.sidebar.selectbox("🧠 Feature Selection Method", list(FS_METHODS.keys()))
run_btn = st.sidebar.button("🚀 Run Prediction")

# ================= MAIN PIPELINE =================
if run_btn:
    dataset_path = DATASETS[dataset]

    X, y, feature_names = preprocess_data(dataset_path)

    # ---- STATIC FEATURE SELECTION ----
    X_sel, selected_features = FS_METHODS[fs_choice](
        X, y, feature_names
    )

    results = train_model(X_sel, y)

    # ================= RESULTS =================
    st.markdown("## 📊 Performance Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", round(results["accuracy"], 3))
    c2.metric("Precision", round(results["precision"], 3))
    c3.metric("Recall", round(results["recall"], 3))
    c4.metric("F1 Score", round(results["f1"], 3))

    st.markdown("## 🧩 Selected Features")
    st.write(selected_features)
    st.write(f"**Total Selected:** {len(selected_features)}")

    st.markdown("## 📉 Confusion Matrix")
    st.pyplot(plot_confusion(results["y_test"], results["y_pred"]))

    st.markdown("## 📈 ROC Curve")
    roc_fig, auc_val = plot_roc(results["y_test"], results["y_prob"])
    st.pyplot(roc_fig)
    st.write(f"**AUC Score:** {auc_val:.3f}")

    st.markdown("## 📋 Comparison Summary")
    st.table(pd.DataFrame([{
        "Dataset": dataset,
        "Feature Selection": fs_choice,
        "Selected Features": len(selected_features),
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1": results["f1"],
        "AUC": auc_val
    }]))

    st.markdown("## 📄 Auto-Generated Report")
    st.write(
        f"""
        The **{dataset} dataset** was evaluated using a **Random Forest classifier**
        with **{fs_choice}-based static feature selection** and **SMOTE**
        for class imbalance handling.

        The hybrid swarm approach automatically identifies the most relevant
        software metrics, reducing dimensionality while improving defect
        prediction accuracy.
        """
    )
