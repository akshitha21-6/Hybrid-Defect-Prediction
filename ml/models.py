from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(X, y):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(Xtr, ytr)

    preds = model.predict(Xte)
    probs = model.predict_proba(Xte)[:, 1]

    return {
        "y_test": yte,
        "y_pred": preds,
        "y_prob": probs,
        "accuracy": accuracy_score(yte, preds),
        "precision": precision_score(yte, preds, zero_division=0),
        "recall": recall_score(yte, preds, zero_division=0),
        "f1": f1_score(yte, preds, zero_division=0),
    }