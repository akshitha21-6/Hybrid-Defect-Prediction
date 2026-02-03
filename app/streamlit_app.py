# ================= MAIN PIPELINE =================
if run_btn:
    dataset_path = DATASETS[dataset]

    # Load data
    X, y, feature_names = preprocess_data(dataset_path)

    # ---- STATIC FEATURE SELECTION ----
    X_sel, selected_features = FS_METHODS[fs_choice](
        X, y, feature_names
    )

    # Train model
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
        with **{fs_choice} static feature selection** and **SMOTE**
        for class imbalance handling.

        Static swarm-based feature selection automatically determines
        the most relevant software metrics, reducing dimensionality
        while improving defect prediction accuracy.
        """
    )
