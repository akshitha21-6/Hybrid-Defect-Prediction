from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_model(model_name):
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

    elif model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss"
        )

    elif model_name == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )