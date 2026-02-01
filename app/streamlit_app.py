import streamlit as st
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

from ml.data_loader import load_data
from ml.model_trainer import get_model
from ml.evaluator import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall,
    plot_cross_validation
)

st.set_page_config(page_title="Hybrid Defect Prediction", layout="wide")

st.title("🚀 Hybrid Software Defect Prediction System")

st.sidebar.header("⚙️ Configuration")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "SVM"]
)

use_smote = st.sidebar.checkbox("Apply SMOTE", value=True)

uploaded_file = st.file_uploader(
    "Upload Defect Dataset (CSV)",
    type=["csv"]
)

if uploaded_file:
    X, y = load_data(uploaded_file)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    model = get_model(model_name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Metrics", "🧮 Confusion Matrix", "📉 Curves", "🔁 Cross Validation"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

    with tab2:
        st.pyplot(plot_confusion_matrix(y_test, y_pred))

    with tab3:
        col1, col2 = st.columns(2)
        col1.pyplot(plot_roc_curve(y_test, y_prob))
        col2.pyplot(plot_precision_recall(y_test, y_prob))

    with tab4:
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="accuracy"
        )
        st.pyplot(plot_cross_validation(cv_scores))
        st.write("Mean CV Accuracy:", np.mean(cv_scores))